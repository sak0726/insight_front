# main.py - 製造図面解析システム v3.0 統合版

#import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Tuple
import asyncio
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import logging
import base64
import threading

print("🚀 insight.py ロード開始")
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

faiss_rebuild_state = {
    "running": False,
    "pending": False,
    "last_started_at": None,
    "last_finished_at": None,
    "last_status": "idle",
    "last_error": None,
    "last_requested_at": None,
    "last_requested_by": None,
    "last_requested_path": None,
}
_faiss_rebuild_lock = threading.Lock()
_faiss_rebuild_worker_running = False
_faiss_rebuild_last_request_ts = 0.0
_FAISS_REBUILD_DEBOUNCE_SEC = 3.0

    
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        body = await request.body()
        logger.error(f"422 validation error. Body: {body.decode('utf-8')}")
    except Exception as e:
        logger.error(f"422 validation error. Failed to read body: {e}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.post("/warmup")
async def warmup(purpose: str = "common"):
    print("🧠 モデルウォームアップ")
    global intelligent_analyzer, supabase
    from supabase import create_client
    try:
        if os.getenv("K_SERVICE") is None:
            try:
                from dotenv import load_dotenv
                load_dotenv(".env")
            except Exception:
                pass
        if intelligent_analyzer is None:
            import importlib
            module = importlib.import_module("intelligent_pdf_analyzer")
            IntelligentPDFAnalyzer = module.IntelligentPDFAnalyzer
            print("🧠 Loading AI Models (IntelligentPDFAnalyzer)...")
            #from intelligent_pdf_analyzer import IntelligentPDFAnalyzer
            intelligent_analyzer = IntelligentPDFAnalyzer()

        if purpose == "search":
            import faiss
            import faiss.contrib.torch_utils

            print("🧠 Loading CPU CLIP model...")
            get_cpu_clip()

            intelligent_analyzer.load_faiss_shard("001")
            intelligent_analyzer.refresh_delta_index()
            print("🧠 Search engine ready")

        logger.info("database load....")
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

    except Exception as e:
        logger.error(f"❌ ウォームアップエラー: {e}")
        raise HTTPException(status_code=500, detail=f"ウォームアップエラー: {str(e)}")


def is_production():
    return os.getenv("K_SERVICE") is not None

from pydantic import BaseModel
# 🚀 メインAPIエンドポイント
# ========================
from pathlib import Path
    
@app.get("/")
def health():
    return {"status": "ok"}

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


class UpdateRequest(BaseModel):
    revision_id: str
    field: str
    value: Optional[Any] = None
    index: Optional[int] = None
    action: Optional[str] = "update"


class DeltaRepairRequest(BaseModel):
    drawing_uid: str


class DeleteDrawingRequest(BaseModel):
    revision_id: Optional[str] = None
    drawing_uid: Optional[str] = None
    rebuild_faiss: Optional[bool] = False


def _faiss_rebuild_worker():
    global _faiss_rebuild_worker_running

    while True:
        while True:
            with _faiss_rebuild_lock:
                pending = bool(faiss_rebuild_state.get("pending"))
                last_req = _faiss_rebuild_last_request_ts
            if not pending:
                with _faiss_rebuild_lock:
                    _faiss_rebuild_worker_running = False
                return

            wait_sec = _FAISS_REBUILD_DEBOUNCE_SEC - (time.time() - last_req)
            if wait_sec <= 0:
                break
            time.sleep(min(wait_sec, 0.5))

        with _faiss_rebuild_lock:
            faiss_rebuild_state["pending"] = False
            faiss_rebuild_state["running"] = True
            faiss_rebuild_state["last_started_at"] = datetime.now(timezone.utc).isoformat()
            faiss_rebuild_state["last_status"] = "running"
            faiss_rebuild_state["last_error"] = None

        try:
            import importlib
            module = importlib.import_module("intelligent_pdf_analyzer")
            module.rebuild_faiss()
            with _faiss_rebuild_lock:
                faiss_rebuild_state["last_status"] = "ok"
        except Exception as e:
            logger.error(f"❌ rebuild_faiss background task error: {e}", exc_info=True)
            with _faiss_rebuild_lock:
                faiss_rebuild_state["last_status"] = "error"
                faiss_rebuild_state["last_error"] = str(e)
        finally:
            with _faiss_rebuild_lock:
                faiss_rebuild_state["running"] = False
                faiss_rebuild_state["last_finished_at"] = datetime.now(timezone.utc).isoformat()


def _request_faiss_rebuild(source: Optional[str] = None, path: Optional[str] = None):
    global _faiss_rebuild_worker_running, _faiss_rebuild_last_request_ts

    with _faiss_rebuild_lock:
        faiss_rebuild_state["pending"] = True
        faiss_rebuild_state["last_requested_at"] = datetime.now(timezone.utc).isoformat()
        faiss_rebuild_state["last_requested_by"] = source
        faiss_rebuild_state["last_requested_path"] = path
        _faiss_rebuild_last_request_ts = time.time()

        if _faiss_rebuild_worker_running:
            return

        _faiss_rebuild_worker_running = True

    thread = threading.Thread(target=_faiss_rebuild_worker, daemon=True)
    thread.start()


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
    import numpy as np
    import cv2
    from intelligent_pdf_analyzer import auto_rotate_image
    try:
        contents = await pdf.read()
        # メモリからPDFを開く
        doc = fitz.open(stream=contents, filetype="pdf")
        TARGET = 1024
        SCALE = 2.0
        DPI = 150 
        pages_info = []

        for i in range(len(doc)):
            page = doc[i]

            # --- A. ページを1024基準で描画（ラスタライズ） ---
            zoom = (TARGET / page.rect.width) * SCALE
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, dpi=DPI, alpha=False)

            img = np.frombuffer(
                pix.samples,
                dtype=np.uint8
            ).reshape(pix.h, pix.w, pix.n)

            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # ② ここで角度取得（※回転しないのが推奨）
            #angle = detect_angle(img)          # ← 推奨
            img, angle = auto_rotate_image(img)
            img_bytes = cv2.imencode(
                ".jpg",
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )[1].tobytes()

            h, w = img.shape[:2]
            
            # --- B. 画像PDFとして1ページPDFを再生成 ---
            new_pdf = fitz.open()
            rect = fitz.Rect(0, 0, w, h)
            p = new_pdf.new_page(width=w, height=h)
            p.insert_image(rect, stream=img_bytes)

            pdf_bytes = new_pdf.tobytes(
                garbage=4,
                deflate=True,
                clean=True
            )
            new_pdf.close()


            b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            pages_info.append({
                "page": i + 1,
                "base64": f"data:application/pdf;base64,{b64_pdf}",
                "img": f"data:image/jpeg;base64,{b64_img}",
            })
        
        doc.close()

        return {"status": "ok", "pages": pages_info}
    
    except Exception as e:
            print(f"Split Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/save")
async def save_files(pdf: List[UploadFile] = File(...), img64:List[UploadFile] = File(...), images: Optional[List[UploadFile]] = File(None), totalSets: int = Form(1), batchId: str = Form(...)):
    print ("🚀 /保存添付ID:", batchId)
    import re
    try:
        # 1. データのメモリ読み込み (非同期処理)
        if not pdf:
            raise HTTPException(status_code=400, detail="PDF is missing")
        
        # フロントが1枚ずつ送ってくるので、先頭の1枚を取得
        target_pdf = pdf[0]
        pdf_bytes = await target_pdf.read()
        img_bytes = await img64[0].read()
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
            img_bytes=img_bytes,
            parts_data=parts_data,
            total_pages=totalSets,
            batch_id=batchId
        )
        status = result.get("status")
        if not status or status != "ok":
            raise Exception("R2キーの取得に失敗しました。")
        elif status == "ocr_test":
            return {"status": "ocr_test", "ocr_result": result.get("ocr_result")}
        return {"status": "ok"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ OCR,labeling Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR,labeling Error: {str(e)}")
@app.get("/batch_progress/{batch_id}")
async def batch_progress(batch_id: str):
    from insight_db import supabase, _execute_with_retry

    def _count_by_status(status: Optional[str] = None) -> int:
        def _op():
            q = (
                supabase
                .table("drawing_batch_items")
                .select("base_dir", count="exact")
                .eq("batch_id", batch_id)
            )
            if status:
                q = q.eq("status", status)
            return q.execute()

        resp = _execute_with_retry(_op, retries=5, base_sleep=0.3)
        return int(resp.count or 0)

    total = _count_by_status()
    queued = _count_by_status("queued")
    processing = _count_by_status("processing")
    done = _count_by_status("done")
    failed = _count_by_status("failed")

    return {
        "total": total,
        "queued": queued,
        "processing": processing,
        "done": done,
        "failed": failed,
    }


@app.post("/batch_retry/{batch_id}")
async def batch_retry(batch_id: str, limit: int = 10):
    from insight_db import retry_failed_batch, mark_batch_processing, mark_batch_done, mark_batch_failed
    global intelligent_analyzer

    if intelligent_analyzer is None:
        import importlib
        module = importlib.import_module("intelligent_pdf_analyzer")
        IntelligentPDFAnalyzer = module.IntelligentPDFAnalyzer
        intelligent_analyzer = IntelligentPDFAnalyzer()

    base_keys = retry_failed_batch(batch_id, limit=limit)
    if not base_keys:
        return {"status": "noop", "message": "no failed rows", "batch_id": batch_id}

    base_dirs = base_keys["base_dirs"]
    mark_batch_processing(base_dirs)

    try:
        vector_count = await asyncio.to_thread(
            intelligent_analyzer.save_clip,
            base_keys["revs"],
            base_keys["all_bytes"],
        )
    except Exception as e:
        mark_batch_failed(base_dirs)
        raise HTTPException(status_code=500, detail=f"retry failed: {str(e)}")

    if vector_count:
        mark_batch_done(base_dirs)
        return {
            "status": "ok",
            "batch_id": batch_id,
            "processed": len(base_dirs),
            "vector_count": int(vector_count),
        }

    mark_batch_failed(base_dirs)
    return {
        "status": "failed",
        "batch_id": batch_id,
        "processed": len(base_dirs),
        "vector_count": 0,
    }

def faiss_search(query_vec_torch, top_k=100):
    import faiss
    import faiss.contrib.torch_utils  # torch Tensor 対応を有効化
    from insight_db import get_data

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

    intelligent_analyzer.refresh_delta_index()

    faiss_index = intelligent_analyzer.faiss_index
    items = intelligent_analyzer.vector_mapping
    delta_index = getattr(intelligent_analyzer, "faiss_delta_index", None)
    delta_items = getattr(intelligent_analyzer, "vector_mapping_delta", [])
    logger.info(f"🔍 FAISS base vectors: {len(items)} / delta vectors: {len(delta_items)}")

    # query_vec_torch: torch.Tensor [1, dim], float32, 正規化済み
    scores, ids = faiss_index.search(query_vec_torch, top_k)
    #print(f"FAISS search scores: {scores}, ids: {ids}")

    best_by_drawing = {}

    def accumulate(score_list, id_list, mapping):
        for score, idx in zip(score_list, id_list):
            if idx < 0 or idx >= len(mapping):
                continue

            meta = mapping[idx]
            uid = meta["drawing_uid"]

            adj_score = float(score)
            if meta.get("role") == "main":
                adj_score += 0.01

            prev = best_by_drawing.get(uid)
            if (prev is None) or (adj_score > prev["score"]):
                best_by_drawing[uid] = {
                    "score": adj_score,
                    "best_idx": int(idx),
                    "best_role": meta.get("role"),
                }

    accumulate(scores[0], ids[0], items)
    if delta_index is not None and delta_items:
        delta_scores, delta_ids = delta_index.search(query_vec_torch, top_k)
        logger.info(f"🔍 Delta search size: {len(delta_scores[0])}")
        accumulate(delta_scores[0], delta_ids[0], delta_items)

    results = []
    for uid, info in best_by_drawing.items():
        full_data = get_data(uid)
        if not full_data or not isinstance(full_data, dict):
            continue

        results.append({
            "score": info["score"],
            "drawing_uid": uid,
            "role": "main",
            "base_dir": full_data.get("base_dir"),
            "full_data": full_data,
        })
    return {
        "results": results
    }


def faiss_search_single(query_vec_torch, top_k=10):
    import faiss
    import faiss.contrib.torch_utils
    from insight_db import get_data

    if intelligent_analyzer is None:
        raise HTTPException(503, "System not warmed up")

    intelligent_analyzer.refresh_delta_index()

    faiss_index = intelligent_analyzer.faiss_index
    items = intelligent_analyzer.vector_mapping
    delta_index = getattr(intelligent_analyzer, "faiss_delta_index", None)
    delta_items = getattr(intelligent_analyzer, "vector_mapping_delta", [])

    scores, ids = faiss_index.search(query_vec_torch, top_k)

    # --- 1. drawing_uid 単位でスコア集約 ---
    best_by_drawing = {}
    def accumulate(score_list, id_list, mapping):
        for score, idx in zip(score_list, id_list):
            if idx < 0 or idx >= len(mapping):
                continue

            uid = mapping[idx]["drawing_uid"]
            if uid not in best_by_drawing or score > best_by_drawing[uid]["score"]:
                best_by_drawing[uid] = {"score": float(score)}

    accumulate(scores[0], ids[0], items)
    if delta_index is not None and delta_items:
        delta_scores, delta_ids = delta_index.search(query_vec_torch, top_k)
        accumulate(delta_scores[0], delta_ids[0], delta_items)

    # --- 2. main meta を引く ---
    merged_items = items + (delta_items or [])
    main_meta_by_uid = {
        item["drawing_uid"]: item
        for item in merged_items
        if item.get("role") == "main"
    }

    # --- 3. main だけ返す ---
    results = []
    for uid, info in best_by_drawing.items():
        main_meta = main_meta_by_uid.get(uid)
        if not main_meta:
            continue
        full_data = get_data(uid)
        if not full_data or not isinstance(full_data, dict):
            continue
        results.append({
            "score": info["score"],
            "drawing_uid": uid,
            "base_dir": main_meta["base_dir"],
            "full_data": full_data,
        })

    return {"results": results}

def build_final_results(top_results):
    final_results = []
    sorted_results = sorted(
        top_results.get("results", []),
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True,
    )

    for item in sorted_results:
        base_dir = item["base_dir"]
        raw_score = float(item.get("score", 0.0))
        sim = max(0.0, min(1.0, raw_score))


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

            "pdf": build_public_url(f"{base_dir}/raw.pdf"),
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
    img64: Optional[UploadFile] = File(None),
    searchMode: Optional[str] = Form('manual'),
    hybridMode: Optional[str] = Form('false')
) -> SearchResponse:
    try:
        logger.info(f"🔍 検索リクエスト受信: searchMode={searchMode}, hybridMode={hybridMode}")
        logger.info(f"📁 PDF: {pdf.filename if pdf else 'なし'}")
        logger.info(f"🖼️ 画像: {len(images) if images else 0}件")
        if img64:
            img_bytes = await img64.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            rawimg = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # 🤖 全自動モード（PDF全体解析）
        if searchMode == 'auto_full' and pdf:
            logger.info("🚀 全自動モードで処理開始")
            return await handle_auto_full_search(rawimg)
        
        # ✂️ 手動切り取りモード（従来機能）
        if images and len(images) > 0:
            return await handle_single_part_search(images)
        
        raise HTTPException(status_code=400, detail="検索に必要なファイルが不足しています")
        
    except Exception as e:
        logger.error(f"❌ 検索エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"検索処理でエラーが発生しました: {str(e)}")

import cv2
import numpy as np


async def handle_auto_full_search(rawimg) -> SearchResponse:
    """🤖 FAISS を使った全自動PDF検索（UI完全互換版）"""
    import torch
    from PIL import Image
    #from intelligent_pdf_analyzer import pdf_bytes_to_raw_image
    import cv2
    model, preprocess = get_cpu_clip()
    import importlib

    try:
        logger.info(f"🚀 全自動PDF解析開始")
        img_rgb = cv2.cvtColor(rawimg, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        image = preprocess(img_pil)          # ← ここで Tensor
        if isinstance(image, torch.Tensor) and image.dim() == 3:  # [C,H,W] の場合のみ
            image = image.unsqueeze(0)            # → [1,C,H,W]

        with torch.no_grad():
            vec = model.encode_image(image)

        vec = vec / vec.norm(dim=-1, keepdim=True)
        query_vec = vec.contiguous()

        top_results = faiss_search(query_vec, top_k=30)
        logger.info(f"🎯 FAISS検索結果: {len(top_results)} 件")
        return build_final_results(top_results)


    except Exception as e:
        logger.error(f"❌ 全自動検索エラー: {e}")
        raise HTTPException(status_code=500, detail=f"全自動検索でエラーが発生しました: {str(e)}")

async def handle_single_part_search(images: List[UploadFile]) -> SearchResponse:
    from fastapi import HTTPException
    import torch
    #from intelligent_pdf_analyzer import parts_bytes_image
    import importlib

    module = importlib.import_module("intelligent_pdf_analyzer")
    parts_bytes_image = module.parts_bytes_image
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

        top_results = faiss_search_single(query_vec, top_k=30)
        print(top_results)
        return build_final_results(top_results)


    except Exception as e:
        logger.error(f"❌ 検索失敗: {str(e)}")
        return SearchResponse(success=False, results=[])
    






@app.get("/data_list")
async def get_data_list(limit: int = 25, offset: int = 0):
    from supabase import create_client
    global supabase
    if os.getenv("K_SERVICE") is None:
            try:
                from dotenv import load_dotenv
                load_dotenv(".env")
            except Exception:
                pass
    if supabase is None:

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

    # Step1: drawingsをupdated_at降順でページネーション（軽量）
    drawings_resp = (
        supabase
        .table("drawings")
        .select("drawing_uid, current_revision_id, updated_at")
        .order("updated_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )

    # Step2: 取得したcurrent_revision_idでrevisionを一括取得（PKルックアップ）
    rev_ids = [r["current_revision_id"] for r in drawings_resp.data if r.get("current_revision_id")]
    revs_resp = (
        supabase
        .table("drawing_revisions")
        .select("revision_id, drawing_uid, drawing_number, parts_name, material, surface, thick, width, length, shape, orientation_deg, img_url, pdf_url, tags_json, parts, updated_at")
        .in_("revision_id", rev_ids)
        .execute()
    )
    revs_map = {r["revision_id"]: r for r in revs_resp.data}

    # drawingsの順序を維持しながらrevisionをマージ
    resp_data = []
    for d in drawings_resp.data:
        rev = revs_map.get(d.get("current_revision_id"))
        if rev:
            resp_data.append({"drawing_uid": d["drawing_uid"], "drawing_revisions": [rev]})

    count_resp = (
        supabase
        .table("drawings")
        .select("drawing_uid", count="exact")
        .execute()
    )
    total = count_resp.count
    print(len(resp_data), total)
    items = []
    if resp_data:
        for row in resp_data:
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
                "surface": rev.get("surface"),
                "thick": rev.get("thick"),
                "width": rev.get("width"),
                "length": rev.get("length"),
                "shape": rev.get("shape"),
                "orientation_deg": rev.get("orientation_deg"),
                "preview_url": build_public_url(rev["img_url"]),
                "pdf_url": build_public_url(rev["pdf_url"]),
                "tags_json": rev.get("tags_json"),
                "parts": build_parts_public(rev.get("parts")),
                "updated_at": rev.get("updated_at"),
            })

    return {
        "items": items,
        "total": total,
    }
@app.get("/data-search")
async def data_search(
    query: str = "",
    name: str = "",
    material: str = "",
    shape: str = "",
    thicknessMin: Optional[float] = None,
    thicknessMax: Optional[float] = None,
    widthMin: Optional[float] = None,
    widthMax: Optional[float] = None,
    heightMin: Optional[float] = None,
    heightMax: Optional[float] = None,
    limit: int = 25,
    offset: int = 0,
):
    global supabase

    has_query = str(query or "").strip() != ""
    has_detail = any([
        str(name or "").strip() != "",
        str(material or "").strip() != "",
        str(shape or "").strip() != "",
        thicknessMin is not None,
        thicknessMax is not None,
        widthMin is not None,
        widthMax is not None,
        heightMin is not None,
        heightMax is not None,
    ])

    if not has_query and not has_detail:
        return {
            "items": [],
            "total": 0,
        }

    if os.getenv("K_SERVICE") is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(".env")
        except Exception:
            pass

    if supabase is None:
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")

        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.warning("SUPABASE env vars are not set. Supabase disabled.")
            supabase = None
            return JSONResponse(
                status_code=200,
                content={
                    "status": "partial",
                    "message": "Models loaded, Supabase disabled",
                },
            )

        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    like_query = f"%{query}%" if has_query else None
    like_name = f"%{name}%" if str(name or "").strip() != "" else None
    like_material = f"%{material}%" if str(material or "").strip() != "" else None
    like_shape = f"%{shape}%" if str(shape or "").strip() != "" else None

    def apply_filters(builder):
        if like_query:
            builder = builder.ilike("drawing_number", like_query)
        if like_name:
            builder = builder.ilike("parts_name", like_name)
        if like_material:
            builder = builder.ilike("material", like_material)
        if like_shape:
            builder = builder.ilike("shape", like_shape)

        if thicknessMin is not None:
            builder = builder.gte("thick", thicknessMin)
        if thicknessMax is not None:
            builder = builder.lte("thick", thicknessMax)
        if widthMin is not None:
            builder = builder.gte("width", widthMin)
        if widthMax is not None:
            builder = builder.lte("width", widthMax)
        if heightMin is not None:
            builder = builder.gte("length", heightMin)
        if heightMax is not None:
            builder = builder.lte("length", heightMax)

        return builder

    resp = apply_filters(
        supabase
        .table("drawing_revisions")
        .select(
            """
            revision_id,
            drawing_uid,
            drawing_number,
            parts_name,
            material,
            surface,
            thick,
            width,
            length,
            shape,
            orientation_deg,
            img_url,
            pdf_url,
            tags_json,
            parts,
            updated_at
            """
        )
    )
    resp = (
        resp
        .order("updated_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )

    count_resp = apply_filters(
        supabase
        .table("drawing_revisions")
        .select(
            """
            revision_id
            """,
            count="estimated",
        )
    ).execute()
    total = count_resp.count

    items = []
    if resp.data:
        for rev in resp.data:
            items.append({
                "drawing_uid": rev.get("drawing_uid"),
                "revision_id": rev.get("revision_id"),
                "drawing_number": rev.get("drawing_number"),
                "parts_name": rev.get("parts_name"),
                "material": rev.get("material"),
                "surface": rev.get("surface"),
                "thick": rev.get("thick"),
                "width": rev.get("width"),
                "length": rev.get("length"),
                "shape": rev.get("shape"),
                "orientation_deg": rev.get("orientation_deg"),
                "preview_url": build_public_url(rev.get("img_url")),
                "pdf_url": build_public_url(rev.get("pdf_url")),
                "tags_json": rev.get("tags_json"),
                "parts": build_parts_public(rev.get("parts")),
                "updated_at": rev.get("updated_at"),
            })

    return {
        "items": items,
        "total": total,
    }


@app.post("/data_update")
async def data_update(payload: UpdateRequest):
    from supabase import create_client
    global supabase

    if os.getenv("K_SERVICE") is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(".env")
        except Exception:
            pass

    if supabase is None:
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")

        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.warning("SUPABASE env vars are not set. Supabase disabled.")
            supabase = None
            return JSONResponse(
                status_code=200,
                content={
                    "status": "partial",
                    "message": "Models loaded, Supabase disabled",
                },
            )

        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    allowed_fields = {
        "drawing_number",
        "parts_name",
        "material",
        "surface",
        "thick",
        "width",
        "length",
        "shape",
        "orientation_deg",
    }
    tag_fields = {"dimensions", "processing_info"}

    field = payload.field
    value = payload.value
    action = (payload.action or "update").lower()

    if field in tag_fields:
        if payload.index is None:
            raise HTTPException(status_code=400, detail="index is required for tag updates")

        resp = (
            supabase
            .table("drawing_revisions")
            .select("tags_json")
            .eq("revision_id", payload.revision_id)
            .single()
            .execute()
        )
        tags = resp.data.get("tags_json") if resp and resp.data else None
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = None
        if not isinstance(tags, dict):
            tags = {"dimensions": [], "processing_info": []}

        current_list = tags.get(field)
        if not isinstance(current_list, list):
            current_list = []

        index = payload.index
        if action == "delete":
            if 0 <= index < len(current_list):
                current_list.pop(index)
        else:
            while len(current_list) <= index:
                current_list.append("")
            current_list[index] = "" if value is None else str(value)
        tags[field] = current_list

        (
            supabase
            .table("drawing_revisions")
            .update({"tags_json": tags})
            .eq("revision_id", payload.revision_id)
            .execute()
        )

        return {"status": "ok", "tags_json": tags}

    if field not in allowed_fields:
        raise HTTPException(status_code=400, detail="field is not allowed")

    if action == "delete":
        value = None
    elif value == "":
        value = None

    (
        supabase
        .table("drawing_revisions")
        .update({field: value})
        .eq("revision_id", payload.revision_id)
        .execute()
    )

    return {"status": "ok"}


@app.post("/delta_repair")
async def delta_repair(payload: DeltaRepairRequest):
    global intelligent_analyzer
    if intelligent_analyzer is None:
        import importlib
        module = importlib.import_module("intelligent_pdf_analyzer")
        IntelligentPDFAnalyzer = module.IntelligentPDFAnalyzer
        intelligent_analyzer = IntelligentPDFAnalyzer()

    try:
        delta_key = intelligent_analyzer.write_delta_for_drawing(payload.drawing_uid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok", "delta_key": delta_key}


@app.post("/data_delete")
async def data_delete(payload: DeleteDrawingRequest):
    from supabase import create_client
    global supabase

    if not payload.revision_id and not payload.drawing_uid:
        raise HTTPException(status_code=400, detail="revision_id or drawing_uid is required")

    if os.getenv("K_SERVICE") is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(".env")
        except Exception:
            pass

    if supabase is None:
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise HTTPException(status_code=503, detail="Supabase is not configured")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    drawing_uid = payload.drawing_uid
    if not drawing_uid:
        rev_resp = (
            supabase
            .table("drawing_revisions")
            .select("drawing_uid")
            .eq("revision_id", payload.revision_id)
            .limit(1)
            .execute()
        )
        if not rev_resp.data:
            raise HTTPException(status_code=404, detail="revision not found")
        drawing_uid = rev_resp.data[0]["drawing_uid"]

    rev_rows = (
        supabase
        .table("drawing_revisions")
        .select("revision_id,base_dir")
        .eq("drawing_uid", drawing_uid)
        .execute()
    ).data or []

    revision_ids = [r["revision_id"] for r in rev_rows if r.get("revision_id")]
    base_dirs = [r["base_dir"] for r in rev_rows if r.get("base_dir")]

    if revision_ids:
        (
            supabase
            .table("drawing_revision_tags")
            .delete()
            .in_("revision_id", revision_ids)
            .execute()
        )

    if base_dirs:
        (
            supabase
            .table("drawing_batch_items")
            .delete()
            .in_("base_dir", base_dirs)
            .execute()
        )

    (
        supabase
        .table("drawing_revisions")
        .delete()
        .eq("drawing_uid", drawing_uid)
        .execute()
    )
    (
        supabase
        .table("drawings")
        .delete()
        .eq("drawing_uid", drawing_uid)
        .execute()
    )

    deleted_r2_keys = 0
    try:
        import importlib
        module = importlib.import_module("intelligent_pdf_analyzer")
        connect_r2 = module.connect_r2
        delete_prefix_from_r2 = module.delete_prefix_from_r2
        list_keys_from_r2 = module.list_keys_from_r2
        DELTA_PREFIX = module.DELTA_PREFIX

        s3 = connect_r2()
        bucket_name = os.getenv("R2_BUCKET_NAME")
        if bucket_name:
            deleted_r2_keys += delete_prefix_from_r2(s3, bucket_name, f"drawings/{drawing_uid}/")
            delta_keys = [k for k in list_keys_from_r2(s3, bucket_name, DELTA_PREFIX) if f"_{drawing_uid}.json" in k]
            if delta_keys:
                s3.delete_objects(
                    Bucket=bucket_name,
                    Delete={"Objects": [{"Key": k} for k in delta_keys]}
                )
                deleted_r2_keys += len(delta_keys)

            if payload.rebuild_faiss:
                _request_faiss_rebuild(source="data_delete", path="/data_delete")
    except Exception as e:
        logger.error(f"❌ R2 delete/rebuild error: {e}", exc_info=True)

    return {
        "status": "ok",
        "drawing_uid": drawing_uid,
        "deleted_revisions": len(revision_ids),
        "deleted_base_dirs": len(base_dirs),
        "deleted_r2_keys": deleted_r2_keys,
        "rebuild_started": bool(payload.rebuild_faiss),
    }


@app.get("/faiss_rebuild_status")
async def faiss_rebuild_status():
    with _faiss_rebuild_lock:
        return dict(faiss_rebuild_state)


@app.post("/faiss_rebuild_request")
async def faiss_rebuild_request(request: Request):
    client_host = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    logger.info(f"🔁 faiss_rebuild_request from {client_host} ua={user_agent}")
    _request_faiss_rebuild(
        source=f"faiss_rebuild_request:{client_host}",
        path=str(request.url.path),
    )
    with _faiss_rebuild_lock:
        return dict(faiss_rebuild_state)

def build_parts_public(parts):
    """
    parts: list[dict] | None
    image_key を public URL に変換して返す
    """
    if not parts or not isinstance(parts, list):
        return []

    out = []
    for p in parts:
        image_key = p.get("image_key")
        if not image_key:
            continue

        out.append({
            "image_key": build_public_url(image_key),
        })

    return out

def build_public_url(path: str) -> str:
    r2_bucket_url = os.getenv("R2_PUBLIC_URL")
    if not r2_bucket_url:
        
        raise HTTPException(503, "R2 is not configured")
    return f"{r2_bucket_url}/{path}"




if __name__ == "__main__":
    print("🚀 insight.py ロード開始")

    import uvicorn

    # app は上で FastAPI() として定義されている想定
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
