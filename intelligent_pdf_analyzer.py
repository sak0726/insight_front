import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import re
import logging
import time
from PIL import Image
from gemiocr import run_gemi
import requests
import base64
import boto3
from io import BytesIO
from pdf2image import convert_from_bytes
# 標準ロギング設定（プロダクション品質）
import logging.handlers
import unicodedata
from botocore.exceptions import ClientError
import faiss
import tempfile
from insight_db import save_ocr_revision, check_batch, save_revision_tags, batch_clear, mark_batch_done, mark_batch_processing, mark_batch_failed

from dotenv import load_dotenv  # ← 追加
load_dotenv()
# ログファイルローテーション設定
log_formatter = logging.Formatter(
    '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
)

# ファイルハンドラー（ローテーション）
file_handler = logging.handlers.RotatingFileHandler(
    'pdf_analyzer.log', maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# コンソールハンドラー
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# ルートロガー設定
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
logger = logging.getLogger(__name__)

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_URL = os.getenv("RUNPOD_URL")
GOOOGLE_API_KEY = os.getenv("GOOOGLE_API_KEY")


def build_filtered_ocr_text(result, img_width, img_height) -> list[dict]:
    ocr_data = []
    min_x_target = img_width * 0#右端から70%
    min_y_target = img_height * 0.7#下端から30%
    if result.read and result.read.blocks:  
        for b in result.read.blocks:
            for line in b.lines:
                bbox = getattr(line, 'bounding_box', None) or getattr(line, 'bounding_polygon', None)
                if bbox and len(bbox) >= 2:
                    
                    try:
                        # 座標取得 (ImagePointオブジェクト対応)
                        if hasattr(bbox[0], 'x'):
                                x_ocr = bbox[0].x
                                y_ocr = bbox[0].y
                        else:
                                x_ocr = float(bbox[0])
                                y_ocr = float(bbox[1])
                    except Exception:
                        continue

                    # 判定: 単純にエリア内かどうか
                    if x_ocr > min_x_target and y_ocr > min_y_target:
                        item = {
                            "text": line.text,
                            "x": int(x_ocr),
                            "y": int(y_ocr)
                        }
                        ocr_data.append(item)
    
    return ocr_data

def crop_to_content(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 2. 2値化 (君の設定を採用: 250で薄い汚れ以外は全部拾う)
    # 反転させることで「描画部分」を白(255)にする
    _, th = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # 3. 輪郭抽出
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img  # 何もなければそのまま

    # 4. 【重要】「ある程度の大きさがある輪郭」だけを集める
    # 小さすぎる点（スキャンのゴミなど）を除外。面積100以下はゴミとみなす
    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]

    if not valid_contours:
        return img # 有効なものがなければそのまま

    # 5. 有効な輪郭「すべて」を含む矩形を計算
    # np.vstack で全ての輪郭座標を合体させてから boundingRect をとる
    x, y, w, h = cv2.boundingRect(np.vstack(valid_contours))

    # 6. 余白（パディング）を追加 (君の5%ロジックを採用)
    h_img, w_img = img.shape[:2]
    pad_w = int(w * 0.05)
    pad_h = int(h * 0.05)
    
    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    w = min(w_img - x, w + pad_w * 2)
    h = min(h_img - y, h + pad_h * 2)

    return img[y:y+h, x:x+w]
def auto_rotate_image(img: np.ndarray) -> tuple[np.ndarray, int]:
    if img is None:
        return img, 0

    # 1. 判定用の「一時的な」縮小画像を作成 (高速化のため)
    #    ※ OpenCLIPやOCRに送る元画像(img)はリサイズしません
    h, w = img.shape[:2]
    scale = 512 / max(h, w)
    
    # 元画像が小さい場合はそのまま使う
    if scale < 1.0:
        small = cv2.resize(img, None, fx=scale, fy=scale)
    else:
        small = img
    
    # 2. グレースケール & 2値化（判定用）
    if len(small.shape) == 3:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    else:
        gray = small
    
    # 文字や線を白(255)にする
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)

    scores = {}
    
    for angle in [0, 90, 180, 270]:
        # -----------------------------
        # 回転（判定用 binary）
        # -----------------------------
        if angle == 0:
            r_img = binary
        elif angle == 90:
            r_img = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            r_img = cv2.rotate(binary, cv2.ROTATE_180)
        elif angle == 270:
            r_img = cv2.rotate(binary, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h_r, w_r = r_img.shape

        # =====================================================
        # 評価A: 横方向の文字・寸法線密度（90/270 排除用）
        # =====================================================
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal_lines = cv2.morphologyEx(r_img, cv2.MORPH_OPEN, kernel_h)
        score_horizontal = cv2.countNonZero(horizontal_lines)

        # =====================================================
        # 評価B: 下部全体の情報密度（表題欄位置ゆらぎ対応）
        # =====================================================
        bottom = r_img[int(h_r * 0.7):, :]
        top = r_img[:int(h_r * 0.3), :]
        score_density = cv2.countNonZero(bottom) - cv2.countNonZero(top)

        # =====================================================
        # 評価C: 図面枠の縦横比（補助）
        # =====================================================
        aspect_ratio = w_r / h_r  # 正方向は >1 になりやすい

        rb = r_img[int(h_r * 0.6):, int(w_r * 0.6):]
        score_rb = cv2.countNonZero(rb)
        # =====================================================
        # 総合スコア（重み確定）
        # =====================================================
        final_score = (
            score_density * 2.5 +
            score_horizontal * 1.5 +
            aspect_ratio * 1.0 +
            score_rb * 3.0      # ★ 右下重視（最重要）
        )

        scores[angle] = final_score

    # 4. ベストな角度を決定
    best_angle = max(scores.items(), key=lambda x: x[1])[0]
    #logger.info(f"🧭 回転スコア: {scores} -> Selected: {best_angle}")
    if best_angle == 0:
        out_img = img
    elif best_angle == 90:
        out_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif best_angle == 180:
        out_img = cv2.rotate(img, cv2.ROTATE_180)
    elif best_angle == 270:
        out_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        out_img = img

    #ok, buf = cv2.imencode(".jpg", out_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return out_img, best_angle


def detect_display_orientation(img: np.ndarray) -> int:
    """
    投影プロファイル法で図面の表示用正位置角度を検出する。

    auto_rotate_image とは独立して動作し、DB保存用の orientation_deg に使用する。
    CLIPベクトル生成・FAISSインデックスには一切影響しない（表示専用）。

    Returns:
        int: 正位置にするために必要な回転角度 (0, 90, 180, 270)
             0=そのまま正位置, 90=時計回り90°で正位置, ...
    """
    if img is None:
        return 0

    # 判定用縮小画像（512px, 高速化）
    h, w = img.shape[:2]
    scale = 512 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(img, None, fx=scale, fy=scale)
    else:
        small = img

    # グレースケール & 2値化（文字・線を白に）
    if len(small.shape) == 3:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    else:
        gray = small
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0
    best_score = -1.0

    for angle in [0, 90, 180, 270]:
        if angle == 0:
            r = binary
        elif angle == 90:
            r = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            r = cv2.rotate(binary, cv2.ROTATE_180)
        else:  # 270
            r = cv2.rotate(binary, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h_r, w_r = r.shape

        # =====================================================
        # 評価1: 水平投影プロファイルの分散（メイン）
        # テキスト行が水平なら行ごとのピクセル数に明確なピーク列が生まれ分散が高くなる
        # =====================================================
        row_sums = np.sum(r, axis=1).astype(np.float64)
        profile_variance = np.var(row_sums)

        # =====================================================
        # 評価2: 下部密度（補助）
        # 表題欄は正位置のとき下部に集中する
        # =====================================================
        bottom_density = float(np.sum(r[int(h_r * 0.65):, :]))
        top_density    = float(np.sum(r[:int(h_r * 0.35), :]))
        score_bottom   = bottom_density - top_density

        score = profile_variance * 1.0 + score_bottom * 0.05

        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle


def image_to_ocr_pdf(img: np.ndarray, jpeg_quality=85) -> bytes:
    import fitz
    import cv2

    # JPEG圧縮
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    ok, jpg = cv2.imencode(".jpg", img, encode_param)
    if not ok:
        raise RuntimeError("JPEG encode failed")

    h, w = img.shape[:2]
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    page.insert_image(page.rect, stream=jpg.tobytes())

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if value is None:
        return ""
    return value


# -----------------------------
# PDF bytes → 低解像度画像
# -----------------------------
def image_to_lowres(img: np.ndarray, max_width=1600):
    h, w = img.shape[:2]
    scale = min(1.0, max_width / w)
    if scale < 1.0:
        img_low = cv2.resize(img, (int(w*scale), int(h*scale)))
    else:
        img_low = img
    return {
        "image": img_low,
        "scale": scale
    }

def pdf_bytes_to_lowres_images(pdf_bytes, max_width=600):
    pages = convert_from_bytes(pdf_bytes, dpi=150)
    results = []

    for p in pages:
        img = np.array(p)
        h, w = img.shape[:2]

        if w > max_width:
            scale = max_width / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            scale = 1.0

        results.append({
            "image": img,
            "scale": scale,
            "orig_size": (h, w)
        })

    return results


# 右下固定クロップ（最優先）
def fixed_title_block_crop(img, w_ratio=0.45, h_ratio=0.30):
    h, w = img.shape[:2]
    cw = int(w * w_ratio)
    ch = int(h * h_ratio)
    x = w - cw
    y = h - ch
    return (x, y, cw, ch)


# フォールバック拡張（OCR薄い用）
def fixed_title_block_crop_large(img, w_ratio=0.55, h_ratio=0.35):
    h, w = img.shape[:2]
    cw = int(w * w_ratio)
    ch = int(h * h_ratio)
    x = w - cw
    y = h - ch
    return (x, y, cw, ch)

def bottom_band_crop(img, h_ratio=0.35):
    h, w = img.shape[:2]
    y = int(h * (1 - h_ratio))
    return (0, y, w, h - y)
# メイン前処理（空は返さない）
def right_band_crop(img, w_ratio=0.35):
    h, w = img.shape[:2]
    cw = int(w * w_ratio)
    x = w - cw
    return (x, 0, cw, h)

def fast_layout_preprocess(img: np.ndarray):
    h, w = img.shape[:2]
    return [[
        {
            "rect": (0, 0, w, h),
            "role": "full_page",
            "score": 1.0
        }
    ]]

DIM_PATTERN = re.compile(
    r"""
    (
        (?:\d+[-x×])?\s*        # 2-Φ7 / 4xM5 など
        (?:Φ|Ø|R|M|t)?\s*       # 記号
        \d+(?:\.\d+)?           # 数値
        (?:\s*[x×]\s*\d+(?:\.\d+)?)*  # 100x50
        (?:\s*\(.*?\))?         # (内) 等
        |
        (?:深さ|面取り|ザグリ)\s*\d+(?:\.\d+)?
    )
    """,
    re.VERBOSE
)


NOTE_PAT = re.compile(
    r"""
    ^
    (?:\d+[.)]|※)?\s*
    .*
    (?:こと|なきこと|する)
    [。．\.]?
    $
    """,
    re.VERBOSE
)
DETAIL_PATTERN = re.compile(
    r"""
    (?:Φ|Ø|R|M|Rc|\d+[-x×])|
    タップ
    """,
    re.VERBOSE
)
THICK_PATTERN = re.compile(
    r"""
    ^
    t\s*\.?\s*\d+(?:\.\d+)?   # t1.5 / t 12 / t.6
    $
    """,
    re.IGNORECASE | re.VERBOSE
)
OUTER_PATTERN = re.compile(
    r"""
    ^
    \d{2,4}(?:\.\d+)?$
    """,
    re.VERBOSE
)
SYMBOL_ONLY_PAT = re.compile(
    r'^[^0-9A-Za-z\u3040-\u30FF\u4E00-\u9FFF]+$'
)
def normalize_ocr_text(t: str) -> str:
    # 全角 → 半角
    t = unicodedata.normalize("NFKC", t)

    # 空白（全角・半角・NBSP 等）を全除去
    t = re.sub(r"\s+", "", t)

    return t

def build_ocr_text(ocr_by_role: dict):
    lines = []
    #logger.info(f"OCR by role: {ocr_by_role}")
    # 表題欄のみ（最重要）
    for role in ("full_page",):
        lines.extend(ocr_by_role.get(role, []))


    cleaned = []
    outer = []
    thickness = []
    detail = []

    for t in lines:
        t = t.strip()
        if not t:
            continue

        t = t.replace("φ", "Φ")
        t = normalize_ocr_text(t)
        # 注意文は全文対象で除外
        if NOTE_PAT.match(t):
            continue
        if SYMBOL_ONLY_PAT.match(t):
            continue
        
        # 板厚
        if THICK_PATTERN.match(t):
            thickness.append(t)
            continue

        # 外形候補
        if OUTER_PATTERN.match(t):
            outer.append(t)
            continue

        # 詳細寸法
        if DETAIL_PATTERN.search(t):
            detail.append(t)

        # GPT用テキスト（寸法含めて渡す）
        cleaned.append(t)

    return {
        "cleaned": cleaned,
        "outer": outer,
        "thickness": thickness,
        "detail": detail
    }

NUM_PATTERN = re.compile(r"""
    (?<![A-Za-z])           # 英字の直後は除外
    \b
    (\d+(?:\.\d+)?)         # 数値
    \b
""", re.VERBOSE)
def normalize_dims(dims: list[str]) -> list[float]:
    nums = []
    for d in dims:
        d = d.replace(",", "").replace("φ", "").replace("Ø", "")
        for m in NUM_PATTERN.findall(d):
            v = float(m)
            # 現実レンジ制約（mm）
            if 0.3 <= v <= 2000:
                nums.append(v)
    return nums


def max_numeric(dims: list[str]) -> float | None:
    nums = normalize_dims(dims)
    return max(nums) if nums else None

def min_numeric(dims: list[str]) -> float | None:
    nums = normalize_dims(dims)
    return min(nums) if nums else None

def make_part_id(base_key: str, page: int, bbox: tuple[int, int, int, int]) -> str:
    """
    bbox = (x, y, w, h)
    """
    raw = f"{base_key}:{page}:{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def summarize_images(images):
    summary = []
    for i, item in enumerate(images):
        b64 = item["image"]
        summary.append({
            "i": i,
            "id": item["id"],
            "b64_len": len(b64),
            "b64_head": b64[:16],
            "b64_tail": b64[-16:],
        })
    return summary

def parts_bytes_image(image_bytes: bytes) -> Image.Image:
    import cv2
    import numpy as np
    from PIL import Image
    
    # 1. bytes -> OpenCV (BGR)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("画像デコード失敗")
    
    # 2. BGR -> RGB -> PIL Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)        
    
    return pil_img

def pdf_bytes_to_raw_image(
    pdf_bytes: bytes,
    dpi: int = 150
) -> Tuple[np.ndarray, str, int]:

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]

    pix = page.get_pixmap(dpi=dpi)
    img_bytes =pix.samples  #uid作成

    img = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(pix.h, pix.w, pix.n)

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    img, angle = auto_rotate_image(img)
    
    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side > 1024:
        scale = 1024 / max_side
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
    )
    #テスト出力
    ts = int(time.time()*1000)
    cv2.imwrite(f"auto_{ts}_final.jpg", img)

    doc.close()
    return img, hashlib.sha256(img_bytes).hexdigest()[:16], angle

def pdf_bytes_to_sha(img_bytes: bytes) -> str:
    return hashlib.sha256(img_bytes).hexdigest()[:16]

def normalize_labels(ocr_result: dict) -> dict:
    dn = ocr_result.get("drawing_number", {})
    value = dn.get("value")
    conf = dn.get("confidence", 0.0)

    if not value or conf < 0.7:
        return {
            "drawing_key": None,
            "confidence": conf,
            "status": "unreliable"
        }

    clean = value.upper().strip()
    clean = re.sub(r"[^\w\-]", "", clean)

    return {
        "drawing_key": clean,
        "confidence": conf,
        "status": "ok"
    }


def calc_pdf_sha256(pdf_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(pdf_bytes)
    return h.hexdigest()


def decide_drawing_uid(pdf_sha256):
    return  pdf_sha256[:16]



def xbuild_faiss_index_from_vector_items(
    vector_items: list[dict],
    shard_id: str,
    upload_func,
) -> None:
    if not vector_items:
        raise ValueError("vector_items is empty")

    # --------------------------------------------------
    # 0. 順序をここで確定（最重要）
    # --------------------------------------------------
    # offset を必須とし、必ず昇順に並べる
    try:
        vector_items = sorted(vector_items, key=lambda x: x["offset"])
    except KeyError:
        raise RuntimeError("vector_items must include 'offset'")

    # --------------------------------------------------
    # 1. vectors を FAISS 用 ndarray に変換
    # --------------------------------------------------
    vectors = np.vstack(
        [item["vector"] for item in vector_items]
    ).astype("float32")

    dim = vectors.shape[1]
    if dim != 512:
        raise ValueError(f"Invalid vector dim: {dim}")

    # cosine 類似度
    faiss.normalize_L2(vectors)

    # --------------------------------------------------
    # 2. FAISS index 構築
    # --------------------------------------------------
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # --------------------------------------------------
    # 3. mapping.json 構築
    # --------------------------------------------------
    mapping_items = []

    for vector_id, item in enumerate(vector_items):
        image_key = item.get("image_key")
        if not image_key and item["role"] == "main":
            image_key = f'{item["base_dir"]}/fullpage.jpg'

        entry = {
            "vector_id": vector_id,
            "drawing_uid": item["drawing_uid"],
            "role": item["role"],
            "base_dir": item["base_dir"],
            "image_key": image_key,  # ★ main も含める
        }

        if item["role"] == "part":
            entry.update({
                "part_id": item["part_id"],
                "page": item["page"],
                "bbox": item["bbox"],
            })

        mapping_items.append(entry)

    mapping = {
        "version": 1,
        "dim": dim,
        "metric": "cosine",
        "shard_id": shard_id,
        "count": len(mapping_items),
        "items": mapping_items,
    }

    # --------------------------------------------------
    # 4. atomic write
    # --------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        faiss_path = os.path.join(tmp, "faiss.index")
        mapping_path = os.path.join(tmp, "mapping.json")

        faiss.write_index(index, faiss_path)
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

        with open(faiss_path, "rb") as f:
            upload_func(
                f.read(),
                f"index/shards/{shard_id}/faiss.index",
                "application/octet-stream"
            )

        with open(mapping_path, "rb") as f:
            upload_func(
                f.read(),
                f"index/shards/{shard_id}/mapping.json",
                "application/json"
            )

def build_faiss_index_from_vector_items(
    vector_items: list[dict],
    shard_id: str,
    s3,
    bucket_name: str,
):
    # offset 昇順で保証
    vector_items.sort(key=lambda x: x["offset"])

    vectors = np.vstack([v["vector"] for v in vector_items]).astype("float32")

    # L2 normalize
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # faiss.index 保存
    faiss_bytes = bytes(faiss.serialize_index(index))
    faiss_key = f"index/shards/{shard_id}/faiss.index"

    s3.put_object(
        Bucket=bucket_name,
        Key=faiss_key,
        Body=faiss_bytes,
        ContentType="application/octet-stream",
    )

    # mapping.json 保存
    mapping = [
        {k: v for k, v in item.items() if k != "vector"}
        for item in vector_items
    ]

    mapping_key = f"index/shards/{shard_id}/mapping.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=mapping_key,
        Body=json.dumps(mapping, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    logger.info(f"✅ FAISS rebuilt: shard={shard_id}, vectors={len(vector_items)}")


def build_vector_order(vision: dict) -> list[str]:
    ids = [f"{vision['base_dir']}:main"]
    for part in vision.get("parts", []):
        ids.append(part["part_id"])  # ← part_id ではない
    return ids

def assemble_vector_items(
    vectors: list[dict],
    meta_data: dict,
) -> list[dict]:
    """
    RunPod推論結果 + meta_data から
    FAISS投入用 vector_items を組み立てる
    """

    vector_items = []

    # --- main ---
    main_vec = vectors[0]["vector"]
    vector_items.append({
        "offset": 0,
        "drawing_uid": meta_data["drawing_uid"],
        "role": "main",
        "part_id": None,
        "base_dir": meta_data["base_dir"],
        "vector": np.array(main_vec, dtype="float32"),
    })

    # --- parts ---
    parts = meta_data.get("parts", [])
    assert len(parts) == len(vectors) - 1, "parts数とvector数が不一致"

    for i, (part, vec_item) in enumerate(zip(parts, vectors[1:])):
        # 保険：順序保証が崩れたら即落とす
        assert part["part_id"] == vec_item["id"]

        vector_items.append({
            "offset": i ,
            "drawing_uid": meta_data["drawing_uid"],
            "role": "part",
            "part_id": part["part_id"],
            "page": part["page"],
            "bbox": part["bbox"],
            "image_key": part["image_key"],
            "base_dir": meta_data["base_dir"],
            "vector": np.array(vec_item["vector"], dtype="float32"),
        })

    return vector_items
def make_fullpage_jpeg_from_img(rawimg: np.ndarray) -> bytes:
    h, w = rawimg.shape[:2]
    scale = 1024 / max(h, w)
    if scale < 1.0:
        rawimg = cv2.resize(rawimg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ts = int(time.time()*1000)
    cv2.imwrite(f"auto_{ts}_final.jpg", rawimg)

    ok, buf = cv2.imencode(".jpg", rawimg, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")

    return buf.tobytes()


def make_fullpage_jpeg_from_pdf(pdf_bytes: bytes, ts: int) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(dpi=200)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, pix.n
    )
    
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img, angle = auto_rotate_image(img)
    
    h, w = img.shape[:2]
    scale = 1024 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    doc.close()

    if not ok:
        raise RuntimeError("JPEG encode failed")

    return buf.tobytes(), angle

def rotate_pdf_bytes(pdf_bytes: bytes, angle: int) -> bytes:
    if angle == 0:
        return pdf_bytes

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        page.set_rotation((page.rotation + angle) % 360)

    rotated_bytes = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return rotated_bytes

def connect_r2():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("R2_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )


DELTA_PREFIX = "index/shared/delta/"


def delete_prefix_from_r2(s3, bucket_name: str, prefix: str) -> int:
    keys = list_keys_from_r2(s3, bucket_name, prefix=prefix)
    if not keys:
        return 0

    deleted = 0
    chunk = 1000
    for i in range(0, len(keys), chunk):
        batch = keys[i:i + chunk]
        s3.delete_objects(
            Bucket=bucket_name,
            Delete={"Objects": [{"Key": k} for k in batch]}
        )
        deleted += len(batch)
    return deleted


def build_delta_payload(vector_items: list[dict]) -> dict:
    if not vector_items:
        return {"version": 1, "dim": 512, "count": 0, "vectors_b64": "", "items": []}

    vectors = np.vstack([item["vector"] for item in vector_items]).astype("float32")
    faiss.normalize_L2(vectors)

    items = []
    for item in vector_items:
        meta = {k: v for k, v in item.items() if k != "vector"}
        items.append(meta)

    return {
        "version": 1,
        "dim": vectors.shape[1],
        "count": len(items),
        "vectors_b64": base64.b64encode(vectors.tobytes()).decode("utf-8"),
        "items": items,
    }
def rebuild_faiss():
    s3 = connect_r2()
    bucket_name = os.getenv("R2_BUCKET_NAME")

    # 1. drawings/**/index.json を全列挙
    keys = list_keys_from_r2(s3, bucket_name, prefix="drawings/")
    index_keys = [k for k in keys if k.endswith("/index.json")]

    shard_vector_items = []
    global_offset = 0

    # 2. index.json ごとに処理
    for index_key in index_keys:
        index_data = json.loads(
            load_bytes_from_r2(s3, bucket_name, index_key)
        )

        current = index_data["current_revision"]
        vector_bin_key = current["vector_bin"]

        vecs = np.frombuffer(
            load_bytes_from_r2(s3, bucket_name, vector_bin_key),
            dtype=np.float32
        ).reshape(-1, 512)

        for v in index_data["vectors"]:
            shard_vector_items.append({
                "offset": global_offset,
                "drawing_uid": index_data["drawing_uid"],
                "role": v.get("role"),
                "part_id": v.get("part_id"),
                "page": v.get("page"),
                "bbox": v.get("bbox"),
                "image_key": v.get("image_key"),
                "base_dir": current["base_dir"],
                "vector": vecs[v["offset"]],
            })
            global_offset += 1

    # 3. FAISS 再構築（全件）
    build_faiss_index_from_vector_items(
        vector_items=shard_vector_items,
        shard_id="001",
        s3=s3,
        bucket_name=bucket_name,
    )
    deleted = delete_prefix_from_r2(s3, bucket_name, DELTA_PREFIX)
    logger.info(f"✅ Cleared delta files: {deleted}")

def list_keys_from_r2(s3, bucket_name: str, prefix: str) -> list[str]:
    keys = []
    token = None

    while True:
        kwargs = {
            "Bucket": bucket_name,
            "Prefix": prefix,
        }
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)

        for obj in resp.get("Contents", []):
            keys.append(obj["Key"])

        if not resp.get("IsTruncated"):
            break

        token = resp.get("NextContinuationToken")

    return keys

def load_bytes_from_r2(s3, bucket_name: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    return obj["Body"].read()


@dataclass 
class IntelligentPDFAnalyzer:
    """🧠 インテリジェントPDF解析エンジン"""
    def __init__(
        self,
        ocr_engine=None,
        clip_engine=None,
    ):
        self.ocr_engine = ocr_engine
        self.clip_engine = clip_engine
                
        # 環境変数からR2接続情報を取得
        self.bucket_name = os.getenv("R2_BUCKET_NAME", "insight-cw")
        account_id = os.getenv("R2_ACCOUNT_ID")
        access_key = os.getenv("R2_ACCESS_KEY_ID")
        secret_key = os.getenv("R2_SECRET_ACCESS_KEY")

        if not (account_id and access_key and secret_key):
            logger.warning("⚠️ R2 Credentials missing! Cloud storage will not work.")
        
        # S3クライアント互換でR2に接続
        self.s3 = boto3.client(
            service_name="s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self.faiss_delta_index = None
        self.vector_mapping_delta = []
        self.delta_keys = set()

    def load_faiss_shard(self, shard_id: str = "001"):
            import faiss, json, tempfile

            index_key = f"index/shards/{shard_id}/faiss.index"
            mapping_key = f"index/shards/{shard_id}/mapping.json"

            index_bytes = self._load_data_from_r2(index_key)
            logger.info(f"Faiss index size: {len(index_bytes)} bytes")

            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "faiss.index")
                with open(path, "wb") as f:
                    f.write(index_bytes)
                    f.flush()
                    os.fsync(f.fileno())   # ★ ローカルでも必須

                # ★ 必ず close 後に読む
                self.faiss_index = faiss.read_index(path)

            # --- mapping.json ---
            mapping_bytes = self._load_data_from_r2(mapping_key)
            mapping_data = json.loads(mapping_bytes.decode("utf-8"))
            if isinstance(mapping_data, dict) and "items" in mapping_data:
                self.vector_mapping = mapping_data["items"]
            else:
                self.vector_mapping = mapping_data

    def refresh_delta_index(self):
            import faiss, json

            delta_keys = self.list_r2_keys(DELTA_PREFIX)
            new_keys = [k for k in delta_keys if k not in self.delta_keys]
            if not new_keys:
                return

            for key in sorted(new_keys):
                payload = json.loads(self._load_data_from_r2(key).decode("utf-8"))
                dim = payload.get("dim", 512)
                count = payload.get("count", 0)
                vectors_b64 = payload.get("vectors_b64", "")
                items = payload.get("items", [])

                if count == 0 or not vectors_b64 or not items:
                    self.delta_keys.add(key)
                    continue

                vec_bytes = base64.b64decode(vectors_b64)
                vectors = np.frombuffer(vec_bytes, dtype=np.float32).reshape(-1, dim)
                faiss.normalize_L2(vectors)

                if self.faiss_delta_index is None:
                    self.faiss_delta_index = faiss.IndexFlatIP(dim)

                self.faiss_delta_index.add(vectors)
                self.vector_mapping_delta.extend(items)
                self.delta_keys.add(key)

    def list_r2_keys(self, prefix: str) -> list[str]:
        keys = []
        continuation = None

        while True:
            kwargs = {
                "Bucket": self.bucket_name,
                "Prefix": prefix,
            }
            if continuation:
                kwargs["ContinuationToken"] = continuation

            resp = self.s3.list_objects_v2(**kwargs)

            for obj in resp.get("Contents", []):
                keys.append(obj["Key"])

            if resp.get("IsTruncated"):
                continuation = resp.get("NextContinuationToken")
            else:
                break

        return keys

    def upload_bytes_to_r2(self, data_bytes: bytes, key: str, content_type: str):
        """ R2へデータをアップロードする """
        try:
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data_bytes,
                ContentType=content_type
            )
            return True
        except Exception as e:
            logger.error(f"❌ R2 Upload Failed ({key}): {e}")
            return False

    def write_delta_from_index(self, index_key: str) -> str:
        index_bytes = self._load_data_from_r2(index_key)
        if not index_bytes:
            raise RuntimeError(f"index.json not found: {index_key}")

        index_data = json.loads(index_bytes.decode("utf-8"))
        current = index_data.get("current_revision") or {}
        vector_bin_key = current.get("vector_bin")
        if not vector_bin_key:
            raise RuntimeError("vector_bin is missing in index.json")

        vec_bytes = self._load_data_from_r2(vector_bin_key)
        if not vec_bytes:
            raise RuntimeError(f"vector.bin not found: {vector_bin_key}")

        vecs = np.frombuffer(vec_bytes, dtype=np.float32).reshape(-1, 512)

        vector_items = []
        for v in index_data.get("vectors", []):
            vector_items.append({
                "offset": v.get("offset"),
                "drawing_uid": index_data.get("drawing_uid"),
                "role": v.get("role"),
                "part_id": v.get("part_id"),
                "page": v.get("page"),
                "bbox": v.get("bbox"),
                "image_key": v.get("image_key"),
                "base_dir": current.get("base_dir"),
                "vector": vecs[v["offset"]],
            })

        delta_payload = build_delta_payload(vector_items)
        delta_key = f"{DELTA_PREFIX}{int(time.time())}_{index_data.get('drawing_uid')}.json"
        ok = self.upload_bytes_to_r2(
            json.dumps(delta_payload, ensure_ascii=False).encode("utf-8"),
            delta_key,
            "application/json",
        )
        if not ok:
            raise RuntimeError("delta upload failed")

        return delta_key

    def write_delta_for_drawing(self, drawing_uid: str) -> str:
        index_key = f"drawings/{drawing_uid}/index.json"
        return self.write_delta_from_index(index_key)
    



    def save_ocr(self, pdf_bytes: bytes, img_bytes: bytes, parts_data: List, total_pages: int, batch_id: str) -> dict[str, str]:
        print("OCR保存処理開始")
        
        start_time = time.time()
        ts = int(time.time()*1000)
        total_time = start_time
        #rawimg, pdf_sha256, angle = pdf_bytes_to_raw_image(pdf_bytes)
        pdf_sha256 = pdf_bytes_to_sha(img_bytes)
        drawing_uid = decide_drawing_uid(pdf_sha256)
        rawimg = img_bytes

        ocrImg = rawimg
        # Gemini OCR前に表示用正位置へ補正
        _img_np = cv2.imdecode(np.frombuffer(rawimg, np.uint8), cv2.IMREAD_COLOR)
        display_deg = detect_display_orientation(_img_np)
        if display_deg == 90:
            _img_np = cv2.rotate(_img_np, cv2.ROTATE_90_CLOCKWISE)
        elif display_deg == 180:
            _img_np = cv2.rotate(_img_np, cv2.ROTATE_180)
        elif display_deg == 270:
            _img_np = cv2.rotate(_img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if display_deg != 0:
            _, _buf = cv2.imencode(".jpg", _img_np, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            ocrImg = _buf.tobytes()

        try:
            result = run_gemi(ocrImg)
            ocr_result, gemiPrice = result if result else ({}, 0)
        except Exception as e:
            logger.error(f"❌ gemi-OCRエラー: {e}")
            raise
        
        print(f"🧾 OCR:費用{gemiPrice:.3f}円, {ocr_result},秒数: {time.time() - start_time:.2f}s, トータル時間: {time.time() - total_time:.2f}s")
        #return {"status": "ocr_test", "ocr_result": ocr_result}  # TODO: テスト用停止点・削除すること
        start_time = time.time()
        if ocr_result.get("drawing_number"):
            drawing_number = str(ocr_result.get("drawing_number"))
        else:
            drawing_number = f"unknown-{ts}"

        enriched_tags = {
            "dimensions": ocr_result.get("dimensions", []),
            "processing_info": ocr_result.get("processing_info", []),
        }

        logger.info(f"drawing_uid: {drawing_uid}, drawing_number: {drawing_number}")
        base_dir = f"drawings/{drawing_uid}/revisions/{ts}"
        parts = []
        parts_json = []
        for part in parts_data:
            page = part["page"]
            bbox = part["bbox"]  # (x, y, w, h)

            part_id = make_part_id(base_dir, page, bbox)
            p_key = f"{base_dir}/parts/{part_id}.jpg"

            # Encode part image to PNG bytes
            part_bytes = part.get("bytes")
            if part_bytes is not None:
                if self.upload_bytes_to_r2(part_bytes, p_key, "image/jpeg"):
                    parts.append({
                        "part_id": part_id,
                        "page": page,
                        "bbox": bbox,
                        "image_key": p_key
                    })
                    parts_json.append({
                        "part_id": part_id,
                        "b64_image": base64.b64encode(part_bytes).decode("utf-8")
                    })
        revision_id = save_ocr_revision(drawing_uid, ocr_result, drawing_number, enriched_tags, f"{base_dir}/raw.pdf", f"{base_dir}/ocr.pdf", f"{base_dir}/fullpage.jpg", f"{base_dir}/raw.jpg", parts, display_deg)
        save_revision_tags(revision_id, ocr_result)
        print(f"💾 DB保存完了, 秒数: {time.time() - start_time:.2f}s, トータル時間: {time.time() - total_time:.2f}s")
        start_time = time.time()



        logger.info(f"💾 R2アップロード開始")

        rawpdf_key = f"{base_dir}/raw.pdf"
        self.upload_bytes_to_r2(pdf_bytes, rawpdf_key, "application/pdf")

        jpg_key = f"{base_dir}/fullpage.jpg"
        self.upload_bytes_to_r2(
            rawimg,
            jpg_key,
            "image/jpeg"
        )

        logger.info(f"💾 r2図面保存完了: {base_dir},秒数: {time.time() - start_time:.2f}s, トータル時間: {time.time() - total_time:.2f}s")

        start_time = time.time()
        try:
            base_keys = check_batch(batch_id, base_dir, total_pages, parts_json)
            #base_keys = check_batcha("50194477")
            if not base_keys:
                logger.info(f"CLIP対象貯蓄中")
                return {"base_dir": base_dir, "status": "ok"}
            mark_batch_processing(base_keys["base_dirs"])
            clip_ok = False
            try:
                vector_count = self.save_clip(base_keys=base_keys["revs"], all_parts_bytes=base_keys["all_bytes"])
                clip_ok = bool(vector_count)
            except Exception:
                logger.error("❌ save_clip exception", exc_info=True)
                clip_ok = False

            if clip_ok:
                mark_batch_done(base_keys["base_dirs"])
            else:
                mark_batch_failed(base_keys["base_dirs"])
                logger.warning("⚠️ save_clip failed. marked as failed")
            if base_keys["is_last_batch"] and clip_ok:
                logger.info(f"🚀 最後の1件、バッチCLIP処理開始、全{total_pages}件、バッチID: {batch_id}")
                batch_clear(batch_id)
        except Exception:
            logger.error("❌ batch enqueue/clip flow failed", exc_info=True)
            try:
                deleted = delete_prefix_from_r2(self.s3, self.bucket_name, f"{base_dir}/")
                logger.warning(f"🧹 rollback R2 prefix: {base_dir}/ deleted={deleted}")
            except Exception:
                logger.error(f"❌ rollback R2 failed: {base_dir}/", exc_info=True)
            try:
                mark_batch_failed([base_dir])
            except Exception:
                logger.error(f"❌ mark failed fallback error: {base_dir}", exc_info=True)
            raise
        #logger.info(f"🚀 バッチCLIP処理完了、全{len(base_keys)}件、秒数: {time.time() - start_time:.2f}s, トータル時間: {time.time() - total_time:.2f}s")
        return {"base_dir": base_dir, "status": "ok"}
    


    def _r2_exists(self, key: str) -> bool:
        if self.s3 is None:
            raise RuntimeError("S3 client is not initialized")

        try:
            self.s3.head_object(
                Bucket=self.bucket_name,
                Key=key
            )
            return True

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")

            # 正常な「存在しない」
            if error_code in ("404", "NoSuchKey", "NotFound"):
                return False

            # それ以外は異常
            logger.error(f"❌ R2 exists check failed ({key}): {e}")
            raise

    def _load_data_from_r2(self, key: str) -> Optional[bytes]:
        """ R2から指定されたキーのデータを取得するヘルパー """
        if self.s3 is None:
            return None
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            return response["Body"].read()

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code == "NoSuchKey":
                return None  # 正常系（初回・未作成）
            logger.error(f"❌ R2 ClientError ({code}): {key}", exc_info=True)
            raise

        except Exception as e:
            logger.error(f"❌ R2 Unknown Error: {key}", exc_info=True)
            raise

    def save_clip(self, base_keys: List[dict], all_parts_bytes: dict):
        """
        /batch_analyze エンドポイント用: 
        R2上の複数の図面データに対し、CLIPベクトル化をまとめて実行する
        """
        logger.info(f"🚀 Batch analysis started")

        all_images_to_send = []
        all_revisions = []
        start_time = time.time()
        # ----------------------------------------------------
        # Phase 1: R2から全データ（PDF/Parts/Metadata）を読み込み、メモリに集める
        # ----------------------------------------------------
        for rec in base_keys:
            try:
                base_key = rec["base_dir"]
                pdf_url = rec["pdf_url"]
                parts = rec.get("parts", [])
                all_revisions.append(rec)              
                logger.info(f"✅ supadata読み込み完了")
                img_bytes = self._load_data_from_r2(base_key + "/fullpage.jpg")
                if not img_bytes: continue
                all_images_to_send.append({
                    "id": f"{base_key}:main",
                    "image": base64.b64encode(img_bytes).decode("utf-8")
                })
                # C. パーツ画像を読み込む
                for part in parts:
                    vid = part["part_id"]
                    part_bytes = all_parts_bytes.get(vid)
                    if not part_bytes:
                        logger.warning(f"⚠️ メモリ内にパーツ画像が見つかりません: {vid}")
                        continue

                    all_images_to_send.append({
                        "id": vid, # IDを part_id (SHA1) に統一
                        "image": base64.b64encode(part_bytes).decode("utf-8")
                    })
                
            except Exception as e:
                logger.error(f"❌ Batch Load Error for {base_key}: {e}")
                
        if not all_images_to_send:
            logger.warning("No images prepared for batch analysis.")
            return
        # ----------------------------------------------------

        vector_map = self._send_batch_to_runpod(all_images_to_send)
        if not vector_map:
            logger.error("❌ RunPod returned no vectors. Batch analysis failed.")
            return

        print(f" vector_map keys: {list(vector_map.keys())}")
        shard_vector_items: list[dict] = []
        for vision in all_revisions:
            base_key = vision["base_dir"]

            ids = build_vector_order(vision)

            vectors = []
            for vid in ids:
                if vid not in vector_map:
                    logger.error(f"❌ Missing vector: {vid}")
                    continue
                vectors.append(vector_map[vid])

            vec = np.array(vectors, dtype=np.float32)
            bin_bytes = vec.tobytes()

            bin_key = f"{base_key}/vector.bin"
            if not self.upload_bytes_to_r2(bin_bytes, bin_key, "application/octet-stream"):
                raise RuntimeError(f"vector.bin upload failed: {bin_key}")


            index = {
                "drawing_uid": vision["drawing_uid"],
                "current_revision": {
                    "base_dir": vision["base_dir"],
                    "vector_bin": f'{vision["base_dir"]}/vector.bin'
                },
                "vectors": []
            }

            # main
            index["vectors"].append({
                "offset": 0,
                "role": "main",
                "image_key": f'{vision["base_dir"]}/fullpage.jpg'
            })

            # parts
            for i, part in enumerate(vision.get("parts", []), start=1):
                index["vectors"].append({
                    "offset": i,
                    "role": "part",
                    "part_id": part["part_id"],
                    "page": part["page"],
                    "bbox": part["bbox"],
                    "image_key": part["image_key"]
                })

            index_key = f'drawings/{vision["drawing_uid"]}/index.json'
            if not self.upload_bytes_to_r2(
                json.dumps(index, ensure_ascii=False, indent=2).encode("utf-8"),
                index_key,
                "application/json"
            ):
                raise RuntimeError(f"index.json upload failed: {index_key}")

            # --- ★ vector_items 組み立て（drawing単位） ---
            vector_items = assemble_vector_items(
                vectors=[
                    {"id": vid, "vector": vector_map[vid]}
                    for vid in ids
                ],
                meta_data=vision,
            )

            delta_payload = build_delta_payload(vector_items)
            delta_key = f"{DELTA_PREFIX}{int(time.time())}_{vision['drawing_uid']}.json"
            if not self.upload_bytes_to_r2(
                json.dumps(delta_payload, ensure_ascii=False).encode("utf-8"),
                delta_key,
                "application/json"
            ):
                raise RuntimeError(f"delta upload failed: {delta_key}")

            # --- ★ shard 全体に追加 ---
            shard_vector_items.extend(vector_items)

            log = {
                "base_key": base_key,
                "vector_key": bin_key,
                "ids": ids
            }
            log_key = f"index/log/{int(time.time())}_{base_key}.json"
            self.upload_bytes_to_r2(json.dumps(log).encode(), log_key, "application/json")

            #meta_key = f"{base_key}/metadata.json"
            
            #json_bytes = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")
            #self.upload_bytes_to_r2(json_bytes, meta_key, "application/json")
            #logger.info(f"✅ Vector written back to R2: {meta_key}")

        if not shard_vector_items:
            raise RuntimeError("No vectors collected for shard")

        logger.info("🎉 Batch analysis complete.")
        logger.info(f"Total time: {time.time() - start_time:.2f}s")

        vector_count = len(shard_vector_items)
        return vector_count


    def _wait_runpod_completion(self, job_id: str, timeout_sec=300, interval=2):
        import time, requests

        if not RUNPOD_URL:
            raise RuntimeError("RUNPOD_URL is not configured")

        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}"
        }

        # runsync を除いた base URL が必要
        base = RUNPOD_URL.rstrip("/").removesuffix("/run")
        status_url = f"{base}/status/{job_id}"

        deadline = time.time() + timeout_sec

        while time.time() < deadline:
            try:
                r = requests.get(status_url, headers=headers, timeout=30)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                # ネットワークエラー等は少し待ってリトライ
                logger.warning(f"Status check failed, retrying... {e}")
                time.sleep(interval)
                continue

            status = data.get("status")
            
            # ログを出して安心感を得る
            if status != "COMPLETED":
                 logger.debug(f"Job Status: {status}")

            if status == "COMPLETED":
                return data

            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"RunPod failed: {data}")

            time.sleep(interval)

        raise TimeoutError(f"RunPod timeout: {job_id}")

    def _send_batch_to_runpod(self, b64_images: list[dict]) -> dict[str, list[float]]:
        """ RunPodへBase64画像のリストを投げてベクトルリストを受け取る """
        if not RUNPOD_API_KEY or not RUNPOD_URL:
            logger.error("❌ RunPod API Key/URL missing.")
            return {}
        
        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RUNPOD_API_KEY}"
        }
        payload = {
            "input": {
                "images": b64_images
            }
        }
        try:
            rp_response = requests.post(RUNPOD_URL, json=payload, headers=headers, timeout=300) # タイムアウト長め
            rp_response.raise_for_status()
            rp_data = rp_response.json()

            status = rp_data.get("status")

            job_id = rp_data.get("id")

            if not job_id:
                logger.error("❌ No job_id returned from RunPod")
                return {}
            
            logger.info(f"Job submitted. ID: {job_id}")

            # 3. 完了を待機 (Polling)
            final_data = self._wait_runpod_completion(job_id)
            
            # 4. 結果を取り出す
            vectors = final_data.get("output", {}).get("vectors")
            if not vectors:
                 logger.error(f"No vectors in output: {final_data}")
                 return {}

            return {
                item["id"]: item["vector"]
                for item in vectors
            }

        except Exception as e:
            logger.error("❌ RunPod Error", exc_info=True)
            return {}
        
