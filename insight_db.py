import base64
import os
import time
from supabase import create_client, Client
from datetime import datetime, timezone
from dotenv import load_dotenv
from uuid import uuid4

from sympy import true

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# クライアント初期化
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


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
def normalize_zero(v):
    """
    OCR未取得を示す 0 系の値を None に変換
    """
    if v in (0, 0.0, "0", "", None):
        return None
    return v
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
) -> str:
    orientation_deg = map_orientation_deg(tags_data)
    ms = tags_data.get("material_size") or {}

    revision_id = str(uuid4())


    supabase.table("drawings").upsert(
        {
            "drawing_uid": drawing_uid,
            "current_revision_id": revision_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        on_conflict="drawing_uid",
    ).execute()

    record = {
        "revision_id": revision_id,
        "drawing_uid": drawing_uid,
        "drawing_number": normalize_str(drawing_number),
        "parts_name": normalize_str(tags_data.get("part_name")),
        "material": normalize_str(tags_data.get("material")),
        "surface": normalize_surface(tags_data.get("surface_treatment")),
        "shape": normalize_str(tags_data.get("shape_category")),
        "thick": normalize_zero(ms.get("thickness_min")),
        "width": normalize_zero(ms.get("width_max")),
        "length": normalize_zero(ms.get("outer_max")),
        "orientation_deg": orientation_deg,
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

    supabase.table("drawing_revisions").insert(record).execute()

    print(f"🧾 revision insert: {revision_id}")
    return revision_id



def save_revision_tags(revision_id: str, tags_raw: dict):
    record = {
        "revision_id": revision_id,
        "tags_raw": tags_raw,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    supabase.table("drawing_revision_tags").upsert(
        record,
        on_conflict="revision_id"
    ).execute()




def check_batch(batch_id: str, base_dir: str, total_pages: int, parts_bytes_json: list):
    # PK(batch_id, base_dir) 前提：重複は upsert で吸収
    supabase.table("drawing_batch_items").upsert(
        {
            "batch_id": batch_id,
            "base_dir": base_dir,
            "parts_bytes": parts_bytes_json,
        },
        on_conflict="base_dir",
    ).execute()

    resp = (
        supabase
        .table("drawing_batch_items")
        .select("base_dir", "parts_bytes")
        .eq("batch_id", batch_id)
        .execute()
    )

    if len(resp.data) != total_pages:
        return None
    
    base_dirs = [r["base_dir"] for r in resp.data]
    revs = (
        supabase.table("drawing_revisions")
        .select("revision_id, drawing_uid, base_dir, pdf_url, parts")
        .in_("base_dir", base_dirs)
        .execute()
    )

    all_bytes_map = {}
    
    for row in resp.data:
        p_list = row["parts_bytes"] # List
        for p in p_list:
            # { "part_id": "b64..." } -> { "part_id": bytes } に戻して保持
            all_bytes_map[p["part_id"]] = base64.b64decode(p["b64_image"])

    return {
        "revs": revs.data,
        "all_bytes": all_bytes_map
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
    return true

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
