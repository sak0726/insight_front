from google import genai
from google.genai import types
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
}
[注意事項]
-thickness_min:板厚を記載。shape_categoryに応じて読み方を変える。「板金(曲げ有)」→「t=X」「t X」表記を最優先で探す(例:材質欄「SUS304 t=1.5」,寸法線のt値)。「板物(平板)」「角物(フライス)」→t=表記がなければ外形寸法のうち最も小さい値(板厚方向)を記載。丸物は最小断面寸法。見つからない場合は0。
-width_max:部材外形の「幅」。正面図または平面図の水平方向最大寸法(外形の横サイズ)。寸法補助線の数値から読む。見つからない場合は0。
-outer_max:部材外形の「丈/長さ」。図面全体で最も大きい外形寸法値。形鋼は断面でなく全長を記載。見つからない場合は0。
-free_hand_textに関しては以下すべてを満たす場合のみ文字列を返す。なければ null,活字（ゴシック/明朝）ではない,図面注記欄・表題欄・注記番号付き文章は除外,-手書き特有の歪み・傾き・不揃いが視認できる.
-dimensionsに関しては主要な数値のみ。JSONを絶対に壊さないこと.
[カテゴリ]板物(平板),板金(曲げ有),丸物(旋盤),角物(フライス),長尺・形鋼(アングル/パイプ),製缶・組図,その他
"""

    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=700,
        response_mime_type="application/json",
        response_schema=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "drawing_number":    types.Schema(type=types.Type.STRING),
                "part_name":         types.Schema(type=types.Type.STRING),
                "material":          types.Schema(type=types.Type.STRING),
                "surface_treatment": types.Schema(type=types.Type.STRING),
                "material_size": types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "thickness_min": types.Schema(type=types.Type.NUMBER),
                        "width_max":     types.Schema(type=types.Type.NUMBER),
                        "outer_max":     types.Schema(type=types.Type.NUMBER),
                    }
                ),
                "shape_category":  types.Schema(type=types.Type.STRING),
                "customer":        types.Schema(type=types.Type.STRING),
                "dimensions":      types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING)
                ),
                "processing_info": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING)
                ),
                "free_hand_text":  types.Schema(type=types.Type.STRING, nullable=True),
            }
        )
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
        
        PRICE_IN_PER_1M = 0.10  #3.1-flash-lite-preview: 0.25
        PRICE_OUT_PER_1M = 0.40  #3.1-flash-lite-preview: 1.50
        USD_JPY = 155.0
        cost_in_usd = (in_tokens / 1_000_000) * PRICE_IN_PER_1M
        cost_out_usd = (out_tokens / 1_000_000) * PRICE_OUT_PER_1M
        total_usd = cost_in_usd + cost_out_usd
        total_jpy = total_usd * USD_JPY
        print(f"📊 [{used_model}] In:{in_tokens}tok / Out:{out_tokens}tok | ${total_usd:.6f} ({total_jpy:.4f}円)")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} | {used_model} | In:{in_tokens} | Out:{out_tokens} | {total_jpy:.4f}円\n"

        with open("log.txt", "a", encoding="utf-8") as f:
            f.write(log_line)
            
        print(f"📝 Log saved: {total_jpy:.4f}円")
        result_json = response.parsed
        ms = result_json.get("material_size", {})
        if ms.get("width_max", 0) > ms.get("outer_max", 0):
            ms["outer_max"] = ms["width_max"]
        return result_json, total_jpy

    except Exception as e:
        print(f"Gemini Extraction Error: {e}")
        return None