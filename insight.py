# main.py - 製造図面解析システム v3.0 統合版

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Tuple
from supabase import create_client
import asyncio
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import logging
import faiss
import base64
from insight_db import get_data
import re

supabase = None
_clip_model = None
_clip_preprocess = None

def get_cpu_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        from open_clip import create_model_and_transforms
        model, _, preprocess = create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device="cpu"
        )
        model.eval()
        _clip_model = model
        # Extract the actual preprocess function from the tuple
        if isinstance(preprocess, (list, tuple)):
            _clip_preprocess = preprocess[-1]
        else:
            _clip_preprocess = preprocess
    return _clip_model, _clip_preprocess

# アプリ作成
intelligent_analyzer = None
clip_engine = None

    
logger = logging.getLogger(__name__)

# 🚀 FastAPI アプリケーション設定
# ==============================

app = FastAPI()
# CORS設定
cors_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイル配信設定
upload_dir = Path("public/uploads")
upload_dir.mkdir(parents=True, exist_ok=True)
COMPARE_DIR = Path("public/compares") 
COMPARE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory="public/uploads"), name="uploads")

# 📁 output ディレクトリ配信設定（PNG画像用）
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")

@app.post("/warmup")
async def warmup():
    print("🧠 モデルウォームアップ")
    global intelligent_analyzer, supabase
    try:
        if intelligent_analyzer is None:
            print("🧠 Loading AI Models (IntelligentPDFAnalyzer)...")
            from intelligent_pdf_analyzer import IntelligentPDFAnalyzer
            intelligent_analyzer = IntelligentPDFAnalyzer()

        get_cpu_clip()
        intelligent_analyzer.load_faiss_shard("001")

        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")

        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.warning("SUPABASE env vars are not set. Supabase disabled.")
            supabase = None
            return JSONResponse(
                status_code=200,
                content={
                    "status": "partial",
                    "message": "Models loaded, Supabase disabled"
                }
            )

        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        logger.info("🧠 モデルウォームアップ完了")
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": "Models and Supabase ready"
            }
        )


        return JSONResponse(content={
            "status": "ok",
            "message": "モデルウォームアップ完了"
        })

    except Exception as e:
        logger.error(f"❌ ウォームアップエラー: {e}")
        raise HTTPException(status_code=500, detail=f"ウォームアップエラー: {str(e)}")


from pydantic import BaseModel
# 🚀 メインAPIエンドポイント
# ========================
from pathlib import Path
    
@app.get("/")
async def root():
    """🏠 システム情報"""
    return {
        "system": "製造図面解析システム",
        "version": "3.0.0",
        "status": "運用中",
        "environment": "プロダクション" if is_production() else "開発",
        "features": [
            "📄 高精度PDF図面解析",
            "🤖 ML自動タグ分類",
            "🎯 候補選択システム", 
            "⚡ プロダクション品質",
            "🔍 類似図面検索"
        ],
        "endpoints": {
            "analyze": "/api/analyze-pdf",
            "search": "/api/search-drawings",
            "health": "/health",
            "metrics": "/api/metrics"
        }
    }

@app.get("/page_image/{page_number}")
async def get_page_image(page_number: int):
    """📄 ページ画像取得（PNG配信）"""
    try:
        # 🎯 アップロード構造: uploads/folder_name/{folder_name}_fullpage.png
        upload_base = Path("public/uploads")
        if not upload_base.exists():
            raise HTTPException(status_code=404, detail="アップロードディレクトリが見つかりません")
        
        # 最新のフォルダを検索
        folder_dirs = [d for d in upload_base.iterdir() if d.is_dir()]
        if not folder_dirs:
            raise HTTPException(status_code=404, detail="画像フォルダが見つかりません")
        
        # 最新のフォルダを使用（作成時間順）
        latest_folder = max(folder_dirs, key=lambda x: x.stat().st_mtime)
        folder_name = latest_folder.name
        
        # 常に {folder_name}_fullpage.png を返す
        fullpage_path = latest_folder / f"{folder_name}_fullpage.png"
        
        if fullpage_path.exists():
            logger.info(f"📄 PNG画像配信: {fullpage_path}")
            return FileResponse(
                path=str(fullpage_path),
                media_type="image/png",
                headers={"Cache-Control": "max-age=3600"}  # 1時間キャッシュ
            )
        else:
            raise HTTPException(status_code=404, detail=f"{folder_name}_fullpage.png が見つかりません")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 画像配信エラー: {e}")
        raise HTTPException(status_code=500, detail=f"画像配信エラー: {str(e)}")

class SearchResponse(BaseModel):
    """検索レスポンス形式（React UI互換）"""
    results: List[Dict[str, Any]]  # React側が期待するフィールド名
    success: bool

class DataListResponse(BaseModel):
    """データ一覧レスポンス形式（既存Node.js互換）"""
    product: List[Dict[str, Any]]


def get_full_page_images(dir_path: Path) -> List[Path]:
    """🔍 全ページ画像ファイルを取得（全自動検索用）- 切り抜きではなくPDF全体のページ画像"""
    image_files = []
    if not dir_path.exists():
        return image_files
    
    # � 各フォルダからPDF全体のページ画像を取得（連番の切り抜きは除外）
    for folder in dir_path.iterdir():
        if folder.is_dir():
            # PDFファイルがある場合のみ処理
            pdf_files = list(folder.glob("*.pdf"))
            if pdf_files:
                # 🔍 全体PNG画像のみを厳密に検索
                png_files = list(folder.glob("*.png"))
                pdf_name = pdf_files[0].stem
                folder_name = folder.name
                
                # 優先順位1: _fullpage.png（解析時に生成される全体画像）
                fullpage_candidates = [f for f in png_files if f.stem.endswith('_fullpage')]
                if fullpage_candidates:
                    image_files.extend(fullpage_candidates)
                    logger.info(f"📄 全体画像取得: {[f.name for f in fullpage_candidates]}")
                    continue
                
                # 他に全体画像がない場合の警告
                logger.warning(f"⚠️ フォルダ {folder_name} に全体PNG画像が見つかりません")
                logger.warning(f"� 利用可能なPNG: {[f.name for f in png_files]}")
                
                # 連番切り抜き（1.png, 2.png等）は除外
    
    return image_files

def get_all_image_files(dir_path: Path) -> List[Path]:
    """すべての画像ファイルを再帰的に取得"""
    image_files = []
    if not dir_path.exists():
        return image_files
    
    for file_path in dir_path.rglob("*.png"):
        image_files.append(file_path)
    for file_path in dir_path.rglob("*.jpg"):
        image_files.append(file_path)
    for file_path in dir_path.rglob("*.jpeg"):
        image_files.append(file_path)
    
    return image_files

def rotate_image(img, angle):
    import cv2
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def _bottom_ratio_score(img):
    import cv2
    import numpy as np

    """
    下 / 上 の黒密度比を返す。
    テーブルが下に来た角度だけ極端に跳ねるので、
    これで正位置を見抜く。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 線を確実に拾うため180で二値化
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    h = th.shape[0]

    # 上30% / 下30%
    top_band    = th[0:int(h * 0.30), :]
    bottom_band = th[int(h * 0.70):h, :]

    top_black = int(np.sum(top_band > 0))
    bottom_black = int(np.sum(bottom_band > 0))

    return bottom_black / (top_black + 1)

@app.get("/health")
async def health_check():
    """ヘルスチェック（既存Node.js互換）"""
    return JSONResponse(content="ok")

@app.post("/cleanup_temp_folders")

def sanitize(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    return value

def resize_long_edge(img, target=1024):
    import cv2

    h, w = img.shape[:2]
    long = max(h, w)
    scale = target / long
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def crop_to_content(img):
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img  # 中身が薄すぎる場合はそのまま

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # ほんの少しだけ余裕（全方向5%）
    pad = int(min(w, h) * 0.05)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = w + pad * 2
    h = h + pad * 2

    return img[y:y+h, x:x+w]

from tempfile import NamedTemporaryFile
@app.post("/split_pdf")
async def split_pdf(pdf: UploadFile = File(...)):
    #高速メモリ展開
    import fitz

    try:
        contents = await pdf.read()
        # メモリからPDFを開く
        doc = fitz.open(stream=contents, filetype="pdf")
        
        pages_info = []

        for i in range(len(doc)):
            # --- A. 本物の単一ページPDFを作成 (base64用) ---
            single_doc = fitz.open()                 # 空のPDF作成
            single_doc.insert_pdf(doc, from_page=i, to_page=i) # ページをコピー
            pdf_bytes = single_doc.tobytes()         # バイナリ化
            single_doc.close()                       # メモリ解放
            
            b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

            # --- B. プレビュー用画像を作成 (images用) ---
            # ※フロントでサムネ表示を高速化したい場合に使用
            page = doc[i]
            pix = page.get_pixmap(dpi=150)           # 解像度は適宜調整
            img_bytes = pix.tobytes("png")
            b64_img = base64.b64encode(img_bytes).decode("utf-8")

            pages_info.append({
                "page": i + 1,
                # ▼ ここが修正点: 中身もヘッダーもPDFにする
                "base64": f"data:application/pdf;base64,{b64_pdf}",
                
                # ▼ 画像は画像として返す (フロントで使い分ける用)
                "images": f"data:image/png;base64,{b64_img}",
                
                "width": page.rect.width,
                "height": page.rect.height
            })
        
        doc.close()

        return {"status": "ok", "pages": pages_info}
    
    except Exception as e:
            print(f"Split Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/save")
async def save_files(pdf: List[UploadFile] = File(...), images: Optional[List[UploadFile]] = File(None), totalSets: int = Form(1), batchId: str = Form(...)):
    print ("🚀 /保存添付ID:", batchId)
    try:
        # 1. データのメモリ読み込み (非同期処理)
        if not pdf:
            raise HTTPException(status_code=400, detail="PDF is missing")
        
        # フロントが1枚ずつ送ってくるので、先頭の1枚を取得
        target_pdf = pdf[0]
        pdf_bytes = await target_pdf.read()
        page_match = re.search(r"page_(\d+)", target_pdf.filename)
        current_page = int(page_match.group(1)) if page_match else 1
        # 画像パーツがあれば、それもメモリに吸い出す
        parts_data = []
        if images:
            for i, img in enumerate(images):
                content = await img.read()
                parts_data.append({
                    "filename": img.filename,
                    "bytes": content,
                    "page": current_page,
                    "bbox": (0, 0, 0, i)
                })

                
        if intelligent_analyzer is None:
            raise HTTPException(503, "Call /warmup first")
        
        result = await asyncio.to_thread(
            intelligent_analyzer.save_ocr,
            pdf_bytes=pdf_bytes,
            parts_data=parts_data,
            total_pages=totalSets,
            batch_id=batchId
        )
        status = result.get("status")
        if not status or status != "ok":
            raise Exception("R2キーの取得に失敗しました。")

    except Exception as e:
        logger.error(f"❌ OCR,labeling Error: {e}", exc_info=True)

def faiss_search(query_vec_torch, top_k=10):
    import faiss
    import faiss.contrib.torch_utils  # torch Tensor 対応を有効化

    if intelligent_analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="System not warmed up. Call /warmup first."
        )

    if not hasattr(intelligent_analyzer, "faiss_index"):
        raise HTTPException(
            status_code=503,
            detail="FAISS index not loaded."
        )

    faiss_index = intelligent_analyzer.faiss_index
    items = intelligent_analyzer.vector_mapping["items"]

    # query_vec_torch: torch.Tensor [1, dim], float32, 正規化済み
    scores, ids = faiss_index.search(query_vec_torch, top_k)
    print(f"FAISS search scores: {scores}, ids: {ids}")
    results = []
    item_count = len(items)

    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0 or idx >= item_count:
            continue

        meta = items[idx]  # ← vector_id 前提で直接 index 参照
        results.append({
            "score": float(score),
            "drawing_uid": meta["drawing_uid"],
            "role": meta["role"],
            "full_data": get_data(meta["drawing_uid"]),
            "base_dir": meta["base_dir"],
        })
    return {
        "results": results
    }


def faiss_search_single(query_vec_torch, top_k=10):
    import faiss
    import faiss.contrib.torch_utils  # torch Tensor 対応を有効化

    if intelligent_analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="System not warmed up. Call /warmup first."
        )

    if not hasattr(intelligent_analyzer, "faiss_index"):
        raise HTTPException(
            status_code=503,
            detail="FAISS index not loaded."
        )

    faiss_index = intelligent_analyzer.faiss_index
    items = intelligent_analyzer.vector_mapping["items"]

    # query_vec_torch: torch.Tensor [1, dim], float32, 正規化済み
    scores, ids = faiss_index.search(query_vec_torch, top_k)
    print(f"FAISS search scores: {scores}, ids: {ids}")
    results = []
    item_count = len(items)
    seen_uids = set()
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0 or idx >= item_count:
            continue

        meta = items[idx]
        uid = meta["drawing_uid"]
        if uid in seen_uids:
            continue

        full_data = get_data(uid)
        if not full_data or not isinstance(full_data, dict):
            continue
        current_base_dir = full_data.get("base_dir")
        
        results.append({
            "score": float(score),
            "drawing_uid": uid,
            "role": meta["role"],
            "full_data": full_data,
            "base_dir": current_base_dir, # 常に現行のパスを使用
        })
        seen_uids.add(uid)
    return {
        "results": results
    }


def build_final_results(top_results):
    final_results = []

    for item in top_results["results"]:
        base_dir = item["base_dir"]
        sim = item["score"]


        # rotation（存在しなければ 0）
        final_results.append({
            "drawing_number": item["full_data"]["drawing_number"],

            # 詳細ポップアップ用（今は非表示で保持）
            "detail": {
                "revision_id": item["full_data"]["revision_id"],
                "drawing_uid": item["full_data"]["drawing_uid"],
                "parts_name": item["full_data"]["parts_name"],
                "material": item["full_data"]["material"],
                "surface": item["full_data"]["surface"],
                "shape": item["full_data"]["shape"],
                "thick": item["full_data"]["thick"],
                "width": item["full_data"]["width"],
                "length": item["full_data"]["length"],
                "tags": item["full_data"]["tags_json"],
                "updated_at": item["full_data"]["updated_at"],
            },

            "pdf": build_public_url(f"{base_dir}/ocr.pdf"),
            "img": [{
                "imgFile": build_public_url(f"{base_dir}/fullpage.jpg"),
                "similarity": sim,
            }],
            "revolutionary_analysis": {
                "confidence": "FAISS + CLIP（auto_full）",
                "dominant_features": [
                    "CLIP特徴（fullpage）",
                    "FAISS上位候補"
                ],
                "breakdown": {
                    "clip": sim,
                    "geometric": 0,
                    "structure": 0,
                    "type": 0,
                    "final": sim
                }
            }
        })

    return {
        "success": True,
        "results": final_results
    }

@app.post("/search")
async def search_similar_images(
    images: Optional[List[UploadFile]] = File(None),
    pdf: Optional[UploadFile] = File(None),
    searchMode: Optional[str] = Form('manual'),
    hybridMode: Optional[str] = Form('false')
) -> SearchResponse:
    try:
        logger.info(f"🔍 検索リクエスト受信: searchMode={searchMode}, hybridMode={hybridMode}")
        logger.info(f"📁 PDF: {pdf.filename if pdf else 'なし'}")
        logger.info(f"🖼️ 画像: {len(images) if images else 0}件")
        # 🤖 全自動モード（PDF全体解析）
        if searchMode == 'auto_full' and pdf:
            logger.info("🚀 全自動モードで処理開始")
            pdf_bytes = await pdf.read()
            return await handle_auto_full_search(pdf_bytes, pdf.filename)
        
        # ✂️ 手動切り取りモード（従来機能）
        if images and len(images) > 0:
            return await handle_single_part_search(images)
        
        raise HTTPException(status_code=400, detail="検索に必要なファイルが不足しています")
        
    except Exception as e:
        logger.error(f"❌ 検索エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"検索処理でエラーが発生しました: {str(e)}")
    
async def handle_auto_full_search(pdf_bytes: bytes, filename: str) -> SearchResponse:
    """🤖 FAISS を使った全自動PDF検索（UI完全互換版）"""
    import torch
    from PIL import Image
    from intelligent_pdf_analyzer import pdf_bytes_to_raw_image

    model, preprocess = get_cpu_clip()

    try:
        logger.info(f"🚀 全自動PDF解析開始: {filename}")
        # -----------------------
        # 1. PDF → fullpage.png
        # -----------------------
        rawimg, _ = pdf_bytes_to_raw_image(pdf_bytes)

        img_rgb = cv2.cvtColor(rawimg, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        image = preprocess(img_pil)          # ← ここで Tensor
        if isinstance(image, torch.Tensor) and image.dim() == 3:  # [C,H,W] の場合のみ
            image = image.unsqueeze(0)            # → [1,C,H,W]

        with torch.no_grad():
            vec = model.encode_image(image)

        vec = vec / vec.norm(dim=-1, keepdim=True)
        query_vec = vec.contiguous()

        top_results = faiss_search(query_vec, top_k=10)
        logger.info(f"🎯 FAISS検索結果: {top_results} 件")
        return build_final_results(top_results)


    except Exception as e:
        logger.error(f"❌ 全自動検索エラー: {e}")
        raise HTTPException(status_code=500, detail=f"全自動検索でエラーが発生しました: {str(e)}")

async def handle_single_part_search(images: List[UploadFile]) -> SearchResponse:
    from fastapi import HTTPException
    import torch
    from intelligent_pdf_analyzer import parts_bytes_image

    # 1. warmup / モデルロード確認
    if intelligent_analyzer is None:
        raise HTTPException(status_code=503, detail="System not warmed up.")

    if not images:
        raise HTTPException(status_code=400, detail="画像が必要です")

    model, preprocess = get_cpu_clip() 

    try:
        content = await images[0].read()
        img_pil = parts_bytes_image(content)

        image = preprocess(img_pil)  # [C, H, W]
        
        if isinstance(image, torch.Tensor) and image.dim() == 3:
            image = image.unsqueeze(0)  # [1, C, H, W]

        with torch.no_grad():
            vec = model.encode_image(image)

        vec = vec / vec.norm(dim=-1, keepdim=True)
        query_vec = vec.contiguous()

        top_results = faiss_search_single(query_vec, top_k=10)
        print(top_results)
        return build_final_results(top_results)


    except Exception as e:
        logger.error(f"❌ 検索失敗: {str(e)}")
        return SearchResponse(success=False, results=[])
    

def load_ai_features(folder_path: Path) -> Optional[Dict[str, Any]]:

    """指定フォルダの ai_features.json を読み込む"""
    json_path = folder_path / "ai_features.json"
    if not json_path.exists():
        logger.warning(f"⚠️ ai_features.json が存在しません: {json_path}")
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ ai_features.json 読み込みエラー ({json_path}): {e}")
        return None
def cosine_similarity(v1, v2):
    import numpy as np

    """ゼロ除算を回避した安全なコサイン類似度"""
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))    




@app.get("/data_list")
async def get_data_list(limit: int = 20, offset: int = 0):
    resp = (
        supabase
        .table("drawings")
        .select(
            """
            drawing_uid,
            drawing_revisions!drawing_revisions_drawing_uid_fkey!inner(
                revision_id,
                drawing_number,
                parts_name,
                material,
                thick,
                width,
                length,
                shape,
                orientation_deg,
                img_url,
                pdf_url,
                updated_at
            )
            """
        )
        .order("updated_at", desc=True, foreign_table="drawing_revisions")
        .range(offset, offset + limit - 1)
        .execute()
    )

    count_resp = (
        supabase
        .table("drawings")
        .select("drawing_uid", count="exact")
        .execute()
    )
    total = count_resp.count
    print(len(resp.data), total)
    items = []
    if resp.data:
        for row in resp.data:
            revs = row.get("drawing_revisions")
            if not revs or len(revs) == 0:  # リストが空またはNoneの場合を考慮
                continue
            
            # 逆参照の場合はリストで返るため、最新（0番目）を抽出
            rev = revs[0]

            items.append({
                "drawing_uid": row["drawing_uid"],
                "revision_id": rev["revision_id"],
                "drawing_number": rev.get("drawing_number"),
                "parts_name": rev.get("parts_name"),
                "material": rev.get("material"),
                "thick": rev.get("thick"),
                "width": rev.get("width"),
                "length": rev.get("length"),
                "shape": rev.get("shape"),
                "orientation_deg": rev.get("orientation_deg"),
                "preview_url": build_public_url(rev["img_url"]),
                "pdf_url": build_public_url(rev["pdf_url"]),
                "updated_at": rev.get("updated_at"),
            })

    return {
        "items": items,
        "total": total,
    }
    
def build_public_url(path: str) -> str:
    r2_bucket_url = os.getenv("R2_PUBLIC_URL")
    if not r2_bucket_url:
        raise RuntimeError("r2.public_url is not configured")
    return f"{r2_bucket_url}/{path}"
