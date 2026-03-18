import base64
import math
import os
import re
import time
import unicodedata
from supabase import create_client, Client
from datetime import datetime, timezone
from dotenv import load_dotenv
from uuid import uuid4
import httpx


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# クライアント初期化
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def _refresh_supabase_client():
    global supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError, httpx.TransportError)):
        return True
    msg = str(exc)
    return "Server disconnected" in msg or "Connection reset" in msg


def _execute_with_retry(op, retries: int = 3, base_sleep: float = 0.4):
    last_exc = None
    for i in range(retries):
        try:
            return op()
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_error(exc) or i == retries - 1:
                raise
            print(f"⚠️ Supabase transient error. retry {i+1}/{retries}: {exc}")
            time.sleep(base_sleep * (2 ** i))
            _refresh_supabase_client()
    raise last_exc


ORIENTATION_MAP = {
    "correct": 0,
    "right_tilted": 90,
    "upside_down": 180,
    "left_tilted": 270
}
def map_orientation_deg(tags_data: dict) -> int:
    label = tags_data.get("orientation")
    if isinstance(label, str):
        return ORIENTATION_MAP.get(label, 0)
    return 0


NUMERIC_TEXT_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def normalize_db_numeric(v):
    """
    numeric カラムに保存できる値だけを通す。
    例: "1/式" のような OCR 混入文字列は None に落とす。
    """
    if v in (None, "", 0, 0.0, "0"):
        return None

    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    if not isinstance(v, str):
        return None

    text = unicodedata.normalize("NFKC", v).strip()
    if not text:
        return None

    text = text.replace(",", "")
    if not NUMERIC_TEXT_PATTERN.fullmatch(text):
        return None

    num = float(text)
    if math.isnan(num) or math.isinf(num) or num == 0:
        return None
    return num


def normalize_surface(v):
    """
    表面処理の無意味値を None に正規化
    """
    if v in ("-", "", None):
        return None
    return v
def normalize_str(v):
    if not isinstance(v, str):
        return None
    v = v.strip()
    return v if v else None


def save_ocr_revision(
    drawing_uid: str,
    tags_data: dict,
    drawing_number: str,
    enriched_tags: dict,
    pdf_url: str,
    ocr_url: str,
    img_url: str,
    rawimg_url: str,
    parts: list,
    display_deg: int,
) -> str:
    ms = tags_data.get("material_size") or {}

    revision_id = str(uuid4())


    _execute_with_retry(lambda: supabase.table("drawings").upsert(
        {
            "drawing_uid": drawing_uid,
            "current_revision_id": revision_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        on_conflict="drawing_uid",
    ).execute())

    record = {
        "revision_id": revision_id,
        "drawing_uid": drawing_uid,
        "drawing_number": normalize_str(drawing_number),
        "parts_name": normalize_str(tags_data.get("part_name")),
        "material": normalize_str(tags_data.get("material")),
        "surface": normalize_surface(tags_data.get("surface_treatment")),
        "shape": normalize_str(tags_data.get("shape_category")),
        "thick": normalize_db_numeric(ms.get("thickness_min")),
        "width": normalize_db_numeric(ms.get("width_max")),
        "length": normalize_db_numeric(ms.get("outer_max")),
        "orientation_deg": display_deg,
        "tags_json": enriched_tags,
        "pdf_url": pdf_url,
        "ocr_url": ocr_url,
        "img_url": img_url,
        "parts": parts,
        "base_dir": os.path.dirname(pdf_url),
        "rawimg_url": rawimg_url,

        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # NOTE:
    # Transient network errors can occur after the server already committed the row.
    # Retrying with INSERT then causes duplicate PK (revision_id) error.
    # Use UPSERT on revision_id to make retries idempotent.
    _execute_with_retry(lambda: supabase.table("drawing_revisions").upsert(
        record,
        on_conflict="revision_id",
    ).execute())

    print(f"🧾 revision insert: {revision_id}")
    return revision_id



def save_revision_tags(revision_id: str, tags_raw: dict):
    record = {
        "revision_id": revision_id,
        "tags_raw": tags_raw,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    _execute_with_retry(lambda: supabase.table("drawing_revision_tags").upsert(
        record,
        on_conflict="revision_id"
    ).execute())




def check_batch(batch_id: str, base_dir: str, total_pages: int, parts_bytes_json: list):
    # PK(batch_id, base_dir) 前提：重複は upsert で吸収
    _execute_with_retry(lambda: supabase.table("drawing_batch_items").upsert(
        {
            "batch_id": batch_id,
            "base_dir": base_dir,
            "parts_bytes": parts_bytes_json,
            "status": "queued",
        },
        on_conflict="base_dir",
    ).execute())

    total_resp = _execute_with_retry(lambda: (
        supabase
        .table("drawing_batch_items")
        .select("base_dir", count="exact")
        .eq("batch_id", batch_id)
        .execute()
    ))
    total_count = total_resp.count or 0

    pending_resp = _execute_with_retry(lambda: (
        supabase
        .table("drawing_batch_items")
        .select("base_dir", count="exact")
        .eq("batch_id", batch_id)
        .eq("status", "queued")
        .execute()
    ))
    pending_count = pending_resp.count or 0

    send_limit = None
    is_last_batch = False

    if pending_count >= 10:
        send_limit = 10
    elif total_count == total_pages and pending_count > 0:
        send_limit = pending_count
        is_last_batch = True
    else:
        return None
    
    resp = _execute_with_retry(lambda: supabase.rpc(
        "pick_clip_batch",
        {
            "p_batch_id": batch_id,
            "p_limit": send_limit,
        }
    ).execute())

    rows = resp.data
    if not rows:
        return None
    
    base_dirs = [r["base_dir"] for r in rows]
    
    revs = _execute_with_retry(lambda: (
        supabase
        .table("drawing_revisions")
        .select("revision_id, drawing_uid, base_dir, pdf_url, parts")
        .in_("base_dir", base_dirs)
        .execute()
    )).data

    all_bytes_map = {}
    
    for row in rows:
        p_list = row["parts_bytes"] # List
        for p in p_list:
            # { "part_id": "b64..." } -> { "part_id": bytes } に戻して保持
            all_bytes_map[p["part_id"]] = base64.b64decode(p["b64_image"])

    return {
        "revs": revs,
        "all_bytes": all_bytes_map,
        "base_dirs": base_dirs,
        "is_last_batch": is_last_batch
    }


def mark_batch_done(base_dirs: list[str]):
    if not base_dirs:
        return

    _execute_with_retry(lambda: supabase.table("drawing_batch_items") \
        .update({"status": "done"}) \
        .in_("base_dir", base_dirs) \
        .execute()
    )
    print(f"✅ mark_batch_done: {base_dirs}")


def mark_batch_processing(base_dirs: list[str]):
    if not base_dirs:
        return

    _execute_with_retry(lambda: supabase.table("drawing_batch_items") \
        .update({"status": "processing"}) \
        .in_("base_dir", base_dirs) \
        .eq("status", "queued") \
        .execute()
    )
    print(f"🚚 mark_batch_processing: {base_dirs}")


def mark_batch_failed(base_dirs: list[str]):
    if not base_dirs:
        return

    _execute_with_retry(lambda: supabase.table("drawing_batch_items") \
        .update({"status": "failed"}) \
        .in_("base_dir", base_dirs) \
        .execute()
    )
    print(f"❌ mark_batch_failed: {base_dirs}")


def retry_failed_batch(batch_id: str, limit: int = 10):
    failed_rows = (
        supabase
        .table("drawing_batch_items")
        .select("base_dir")
        .eq("batch_id", batch_id)
        .eq("status", "failed")
        .limit(limit)
        .execute()
        .data
    )

    if not failed_rows:
        return None

    base_dirs = [r["base_dir"] for r in failed_rows]

    (
        supabase
        .table("drawing_batch_items")
        .update({"status": "queued"})
        .in_("base_dir", base_dirs)
        .eq("batch_id", batch_id)
        .eq("status", "failed")
        .execute()
    )

    rows = (
        supabase
        .table("drawing_batch_items")
        .select("base_dir, parts_bytes")
        .eq("batch_id", batch_id)
        .in_("base_dir", base_dirs)
        .execute()
        .data
    )

    revs = (
        supabase
        .table("drawing_revisions")
        .select("revision_id, drawing_uid, base_dir, pdf_url, parts")
        .in_("base_dir", base_dirs)
        .execute()
    ).data

    all_bytes_map = {}
    for row in rows:
        p_list = row.get("parts_bytes") or []
        for p in p_list:
            all_bytes_map[p["part_id"]] = base64.b64decode(p["b64_image"])

    return {
        "revs": revs,
        "all_bytes": all_bytes_map,
        "base_dirs": base_dirs,
    }


def extract_search_fields(tags: dict) -> dict:
    ms = tags.get("material_size", {}) or {}

    return {
        "drawing_number": tags.get("drawing_number"),
        "parts_name": tags.get("part_name"),
        "material": tags.get("material"),
        "surface": tags.get("surface_treatment"),
        "thick": ms.get("thickness_min"),
        "width": ms.get("width_max"),
        "length": ms.get("outer_max"),
        "shape": tags.get("shape_category"),
        "tags_json": tags,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

def batch_clear(batch_id: str):
    supabase.table("drawing_batch_items").delete().eq("batch_id", batch_id).execute()
    return True

def get_data(drawing_uid: str):
    resp = (
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
            shape,
            thick,
            width,
            length,
            tags_json,
            base_dir,
            ocr_url,
            img_url,
            orientation_deg,
            updated_at
            """
        )
        .eq("drawing_uid", drawing_uid)
        .limit(1)
        .execute()
    )

    if not resp.data:
        return None

    return resp.data[0]


def get_batch_progress(batch_id: str):
    rows = (
        supabase
        .table("drawing_batch_items")
        .select("status", count="exact")
        .eq("batch_id", batch_id)
        .execute()
    )

    # status別カウント
    stats = {
        "queued": 0,
        "processing": 0,
        "done": 0,
        "failed": 0,
        "total": rows.count or 0
    }

    data = (
        supabase
        .table("drawing_batch_items")
        .select("status")
        .eq("batch_id", batch_id)
        .execute()
        .data
    )

    for r in data:
        s = r["status"]
        if s in stats:
            stats[s] += 1

    return stats
