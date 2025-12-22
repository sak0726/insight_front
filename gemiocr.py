from google import genai
from google.genai import types
import json
import os
import datetime
from dotenv import load_dotenv
import asyncio
load_dotenv()

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def run_gemi(jpeg_bytes):
    prompt = """画像解析し下記JSONのみ出力.MD単位(mm)不要.
{
"drawing_number":"図番,drawing no(表題欄)",
"part_name":"品名(表題欄)",
"material":"材質(例:SUS304,組立品)",
"surface_treatment":"表面処理(例:アルマイト,なし)",
"material_size":{"thickness_min":0,"width_max":0,"outer_max":0},
"shape_category":"カテゴリ(下リストより選択)",
"customer":"発注元(敬称除外)",
"dimensions":["主要寸法,数値のみ,大中小いれて10個程度"],
"processing_info":["ネジ/穴(重複なし。例:M6,2-φ10)","はめあい公差(例:H7)※数値公差/粗さ/JIS規格名除外", 最大20個],
"free_hand_text": "",
"orientation": title_block_position,
}
[title_block_position Judgment Rules]
Identify the physical location of the Title Block
(the table that contains BOTH:
- Company Name
- Drawing Number
- Scale
AND is typically located near the drawing border)

Assume the image coordinate system:
- Origin is at the TOP-LEFT of the image.
- X increases to the RIGHT.
- Y increases DOWNWARD.

Determine which CORNER of the image contains the Title Block.

Output exactly one of the following values:

- "correct"
  → Title Block is located at the BOTTOM-RIGHT corner.

- "upside_down"
  → Title Block is located at the TOP-LEFT corner.

- "right_tilted"
  → Title Block is located at the BOTTOM-LEFT corner.

- "left_tilted"
  → Title Block is located at the TOP-RIGHT corner.

Strict rules:
- Judge by the physical position of the table, not by text orientation.
- Do NOT infer rotation by readability of text.
- If uncertain, choose the closest corner.
- Output ONLY the label string.
- The Title Block is always attached to or very close to the drawing border.
- Tables floating inside the drawing area are NOT Title Blocks.
- Company Name may appear as a logo or abbreviated company text.
- Drawing Number may appear as "DWG No.", "Drawing No.", or similar.
- Scale may appear as "SCALE" or a numeric ratio.
- If multiple tables satisfy the conditions, choose the one closest to the image border.

[注意事項]
-material_sizeに関してはthickness_min:部材の「板厚」(図面内最小値),width_max:部材の「幅」(図面内最大値).
-outer_max重要指示:部材の「切断長」(図面内最大値).形鋼は断面でなく長さを記載.
-free_hand_textに関しては以下すべてを満たす場合のみ文字列を返す。なければ null,活字（ゴシック/明朝）ではない,図面注記欄・表題欄・注記番号付き文章は除外,-手書き特有の歪み・傾き・不揃いが視認できる.
-dimensionsに関しては主要な数値のみ。JSONを絶対に壊さないこと.
[カテゴリ]板物(平板),板金(曲げ有),丸物(旋盤),角物(フライス),長尺・形鋼(アングル/パイプ),製缶・組図,その他
"""

    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=700,
        response_mime_type="application/json"
    )

    try:
        response = client.models.generate_content(
            model='gemini-flash-lite-latest', 
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ],
            config=config
        )
        used_model = getattr(response, "model_version", "Unknown")
        print(f"🤖 [Used Model] {used_model}")

        usage = response.usage_metadata
        in_tokens = usage.prompt_token_count
        out_tokens = usage.candidates_token_count
        
        PRICE_IN_PER_1M = 0.1
        PRICE_OUT_PER_1M = 0.4
        USD_JPY = 155.0
        cost_in_usd = (in_tokens / 1_000_000) * PRICE_IN_PER_1M
        cost_out_usd = (out_tokens / 1_000_000) * PRICE_OUT_PER_1M
        total_usd = cost_in_usd + cost_out_usd
        total_jpy = total_usd * USD_JPY
        print(f"📊 [Gemini Cost] In: {in_tokens} tokens, Out: {out_tokens} tokens")
        print(f"💰 [Gemini Cost] Total: ${total_usd:.6f} ({total_jpy:.4f} 円)")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} | {used_model} | In:{in_tokens} | Out:{out_tokens} | {total_jpy:.4f}円\n"

        with open("log.txt", "a", encoding="utf-8") as f:
            f.write(log_line)
            
        print(f"📝 Log saved: {total_jpy:.4f}円")
        raw_text = response.text
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        result_json = json.loads(cleaned_text)
        if isinstance(result_json  , list):
            result_json = result_json[0] if result_json else {}
            return result_json, total_jpy
        
        return result_json, total_jpy

    except Exception as e:
        print(f"Gemini Extraction Error: {e}")
        return None