import json
import time
import math
import os
import re
import requests
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from dotenv import load_dotenv
import streamlit as st

MODEL_NAME = "gemini-2.5-flash"
USDA_CALORIE_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
DEBUG_RUN_ID = "pre_fix"
DEBUG_LOG_PATH = r"C:\Users\chiac\Desktop\gdgkl-Hackathon\.cursor\debug.log"


def _agent_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any], run_id: str = DEBUG_RUN_ID) -> None:
    # region agent log
    try:
        payload = {
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": time.time() * 1000,
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion

VISION_ANALYSIS_PROMPT = (
    "You are a nutrition assistant. Analyze the uploaded food image and return ONLY "
    "valid JSON with keys:\n"
    "detected_foods (array of up to 5 likely food items),\n"
    "healthy_score (integer 0-100),\n"
    "reason (short summary),\n"
    "improvement_tips (array of up to 5 short tips),\n"
    "fiber_level (one of: Low, Medium, High),\n"
    "sugar_level (one of: Low, Medium, High),\n"
    "protein_g (estimated grams of protein as a number).\n"
    "Respond with concise, realistic health guidance and no markdown."
)

CALORIE_REFINEMENT_PROMPT = (
    "You are a nutrition assistant. You have image analysis results and calorie "
    "lookup data from a food-nutrition API. Return ONLY JSON with keys:\n"
    "detected_foods (array of food names),\n"
    "estimated_total_calories_kcal (number),\n"
    "healthy_score (integer 0-100),\n"
    "reason (short summary),\n"
    "improvement_tips (array of up to 5 short tips),\n"
    "fiber_level (one of: Low, Medium, High),\n"
    "sugar_level (one of: Low, Medium, High),\n"
    "protein_g (estimated grams of protein as a number).\n"
    "No markdown."
)


# ─────────────────────────── session state ───────────────────────────

def init_session_state() -> None:
    if "food_image" not in st.session_state:
        st.session_state.food_image = None
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None


# ─────────────────────────── helpers ─────────────────────────────────

def mime_from_upload(uploaded_file) -> str:
    if uploaded_file and uploaded_file.type:
        return uploaded_file.type
    if not uploaded_file or not uploaded_file.name:
        return "image/jpeg"
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/jpeg")


def extract_json(response_text: str) -> Dict[str, Any]:
    if not response_text:
        raise ValueError("Empty response from Gemini.")
    match = re.search(r"\{[\s\S]*\}", response_text.strip())
    if not match:
        raise ValueError("Could not find JSON content in model response.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model response did not return a JSON object.")
    return parsed


def normalize_result(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    score_raw = raw.get("healthy_score")
    tips_raw = raw.get("improvement_tips")
    reason = raw.get("reason", "No reason provided.")
    foods_raw = raw.get("detected_foods", raw.get("foods", []))

    try:
        score = int(score_raw)
    except (TypeError, ValueError):
        return {}, f"Could not parse healthy_score: {score_raw!r}"

    detected_foods = (
        foods_raw if isinstance(foods_raw, list)
        else [str(foods_raw)] if foods_raw else []
    )
    detected_foods = [str(f).strip() for f in detected_foods if str(f).strip()]
    score = max(0, min(100, score))

    if not isinstance(tips_raw, list):
        tips_raw = [str(tips_raw)] if tips_raw else []
    tips = [str(t).strip() for t in tips_raw if str(t).strip()]
    if not tips:
        tips = ["Improve nutrition quality in future analysis."]

    estimated_calories_raw = raw.get(
        "estimated_total_calories_kcal",
        raw.get("total_estimated_calories_kcal", raw.get("total_calories_kcal")),
    )
    try:
        estimated_calories_kcal = (
            float(estimated_calories_raw) if estimated_calories_raw is not None else None
        )
    except (TypeError, ValueError):
        estimated_calories_kcal = None

    calorie_lookup = raw.get("calorie_lookup", raw.get("calorie_data", []))
    if not isinstance(calorie_lookup, list):
        calorie_lookup = [calorie_lookup] if calorie_lookup else []

    normalized_calories = []
    for item in calorie_lookup:
        if not isinstance(item, dict):
            continue
        ni = {
            "food_name": str(item.get("food_name", item.get("name", ""))).strip(),
            "calories_kcal": item.get("calories_kcal"),
            "matched_name": str(item.get("matched_name", "")).strip(),
            "serving_grams": item.get("serving_grams"),
            "source": str(item.get("source", "calorie_api")).strip() or "calorie_api",
        }
        if ni["food_name"] and ni["calories_kcal"] is not None:
            normalized_calories.append(ni)

    # Nutrient levels
    valid_levels = {"Low", "Medium", "High"}
    fiber_level = str(raw.get("fiber_level", "")).strip().capitalize()
    sugar_level = str(raw.get("sugar_level", "")).strip().capitalize()
    fiber_level = fiber_level if fiber_level in valid_levels else None
    sugar_level = sugar_level if sugar_level in valid_levels else None

    protein_raw = raw.get("protein_g")
    try:
        protein_g = float(protein_raw) if protein_raw is not None else None
    except (TypeError, ValueError):
        protein_g = None

    return {
        "healthy_score": score,
        "reason": str(reason).strip() or "No reason provided.",
        "improvement_tips": tips[:5],
        "detected_foods": detected_foods[:10],
        "estimated_total_calories_kcal": estimated_calories_kcal,
        "calorie_lookup": normalized_calories,
        "fiber_level": fiber_level,
        "sugar_level": sugar_level,
        "protein_g": protein_g,
    }, None


# ─────────────────────────── USDA calorie API ────────────────────────

def fetch_usda_calories(food_name: str, api_key: str) -> Optional[Dict[str, Any]]:
    params = {"query": food_name, "pageSize": 1, "api_key": api_key}
    response = requests.get(USDA_CALORIE_API_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    foods = payload.get("foods", [])
    if not foods:
        return None

    first_food = foods[0]
    nutrients = first_food.get("foodNutrients", [])
    calories = None
    for nutrient in nutrients:
        if not isinstance(nutrient, dict):
            continue
        name = str(nutrient.get("nutrientName", "")).lower()
        nutrient_id = nutrient.get("nutrientId")
        if nutrient_id == 1008 or name == "energy":
            try:
                calories = float(nutrient.get("value"))
            except (TypeError, ValueError):
                calories = None
            break

    if calories is None:
        return None

    return {
        "food_name": food_name,
        "matched_name": first_food.get("description", "").strip(),
        "calories_kcal": calories,
        "serving_grams": first_food.get("servingSize"),
        "source": "usda",
    }


def fetch_calories_for_foods(
    foods: List[str], api_key: str
) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    lookup, total, has_data = [], 0.0, False
    for food in foods:
        food_name = str(food).strip()
        if not food_name:
            continue
        try:
            item = fetch_usda_calories(food_name, api_key)
        except (requests.RequestException, ValueError):
            continue
        if not item:
            continue
        lookup.append(item)
        total += float(item["calories_kcal"])
        has_data = True
    return lookup, (total if has_data else None)


def refine_with_calories(
    api_key: str,
    image_result: Dict[str, Any],
    calorie_lookup: List[Dict[str, Any]],
    estimated_total_calories_kcal: Optional[float],
) -> Dict[str, Any]:
    client = genai.Client(api_key=api_key)
    total_calories_text = (
        str(estimated_total_calories_kcal) if estimated_total_calories_kcal is not None else "N/A"
    )
    context = {
        "vision_result": image_result,
        "calorie_lookup": calorie_lookup,
        "estimated_total_calories_kcal": estimated_total_calories_kcal,
    }
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            CALORIE_REFINEMENT_PROMPT,
            f"Use this context and return JSON now:\n{json.dumps(context, ensure_ascii=False)}",
            f"Total estimated calories: {total_calories_text}",
        ],
    )
    text = getattr(response, "text", "")
    if text is None:
        raise RuntimeError("Gemini did not return response text for calorie-refined analysis.")
    return extract_json(text.strip())


def analyze_food_image(
    api_key: str,
    image_bytes: bytes,
    mime_type: str,
    calorie_api_key: str = "",
) -> Dict[str, Any]:
    # region agent log
    _agent_log(
        hypothesis_id="B",
        location="app.py:analyze_food_image",
        message="starting vision analysis",
        data={
            "api_key_len": len(api_key),
            "mime_type": mime_type,
            "image_bytes_len": len(image_bytes),
            "has_calorie_key": bool(calorie_api_key.strip()),
        },
    )
    # endregion
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            VISION_ANALYSIS_PROMPT,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
    )
    # region agent log
    _agent_log(
        hypothesis_id="E",
        location="app.py:analyze_food_image",
        message="vision response received",
        data={"has_text": bool(getattr(response, "text", "")), "response_type": str(type(response).__name__)},
    )
    # endregion
    text = getattr(response, "text", "")
    if text is None:
        # region agent log
        _agent_log(
            hypothesis_id="D",
            location="app.py:analyze_food_image",
            message="gemini returned no text",
            data={},
        )
        # endregion
        reason = ""
        try:
            if response.candidates:
                reason = str(response.candidates[0].finish_reason)
        except Exception:
            reason = "unknown"
        raise RuntimeError(
            f"Gemini did not return analysis text. Finish reason: {reason or 'unknown'}."
        )

    text = text.strip()
    parsed = extract_json(text)
    normalized, issue = normalize_result(parsed)
    if issue:
        raise ValueError(issue)

    detected_foods = normalized.get("detected_foods", [])
    if not calorie_api_key.strip() or not detected_foods:
        return normalized

    calorie_lookup, total_calories = fetch_calories_for_foods(
        foods=detected_foods, api_key=calorie_api_key.strip()
    )
    if not calorie_lookup:
        return normalized

    try:
        refined = refine_with_calories(
            api_key=api_key,
            image_result=normalized,
            calorie_lookup=calorie_lookup,
            estimated_total_calories_kcal=total_calories,
        )
    except Exception:
        normalized["calorie_lookup"] = calorie_lookup
        normalized["estimated_total_calories_kcal"] = total_calories
        return normalized

    refined_result = refined if isinstance(refined, dict) else {}
    if isinstance(refined, str):
        refined_result = extract_json(refined)

    normalized, issue = normalize_result(refined_result)
    if issue:
        raise ValueError(issue)
    if not normalized.get("calorie_lookup"):
        normalized["calorie_lookup"] = calorie_lookup
    if normalized.get("estimated_total_calories_kcal") is None:
        normalized["estimated_total_calories_kcal"] = total_calories
    return normalized


# ─────────────────────────── CSS injection ───────────────────────────

def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Base & Reset ─────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }
        [data-testid="stAppViewContainer"] {
            background: #F0F4F8;
        }
        [data-testid="stHeader"] { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }
        .block-container {
            padding: 0 0 40px 0 !important;
            max-width: 430px !important;
            margin: 0 auto !important;
        }
        #MainMenu, footer { visibility: hidden; }

        /* ── App Header ───────────────────────────────────────────── */
        .jm-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #ffffff;
            padding: 14px 20px 12px 20px;
            border-bottom: 1px solid #F0F2F5;
            position: sticky;
            top: 0;
            z-index: 200;
        }
        .jm-logo {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .jm-logo-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #6C63FF 0%, #4F46E5 100%);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-weight: 800;
            font-size: 17px;
            letter-spacing: -0.5px;
        }
        .jm-logo-text {
            font-size: 20px;
            font-weight: 700;
            color: #1A1A2E;
            letter-spacing: -0.3px;
        }
        .jm-profile-btn {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #F5F6FA;
            border: 1px solid #E8EAF0;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* ── Page padding wrapper ──────────────────────────────────── */
        .jm-page { padding: 16px 16px 0 16px; }

        /* ── Generic Card ─────────────────────────────────────────── */
        .jm-card {
            background: #ffffff;
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        }
        .jm-card-title {
            font-size: 18px;
            font-weight: 700;
            color: #1A1A2E;
            margin: 0 0 4px 0;
        }
        .jm-card-subtitle {
            font-size: 13px;
            color: #8A8FA8;
            margin: 0 0 16px 0;
            line-height: 1.45;
        }

        /* ── File Uploader override ────────────────────────────────── */
        [data-testid="stFileUploader"] {
            background: transparent !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #4CAF50 !important;
            border-radius: 14px !important;
            background: rgba(76,175,80,0.04) !important;
            padding: 24px 16px !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            background: rgba(76,175,80,0.08) !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] > div > span {
            font-size: 14px !important;
            font-weight: 500 !important;
            color: #4A4F6A !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] > div > small {
            color: #AAB0C4 !important;
            font-size: 12px !important;
        }
        /* Green upload icon color */
        [data-testid="stFileUploaderDropzoneInstructions"] svg {
            color: #4CAF50 !important;
            fill: #4CAF50 !important;
        }

        /* ── Uploaded image preview ───────────────────────────────── */
        [data-testid="stImage"] > img {
            border-radius: 14px !important;
            object-fit: cover;
            max-height: 220px;
            width: 100%;
        }

        /* ── Analyze button ───────────────────────────────────────── */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 13px 20px !important;
            font-size: 15px !important;
            font-weight: 600 !important;
            width: 100% !important;
            box-shadow: 0 4px 14px rgba(46,125,50,0.35) !important;
            letter-spacing: 0.2px !important;
        }
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(46,125,50,0.4) !important;
        }
        .stButton > button[kind="primary"]:disabled {
            background: #C8E6C9 !important;
            box-shadow: none !important;
            cursor: not-allowed !important;
        }
        /* Secondary/clear button */
        .stButton > button[kind="secondary"] {
            background: #F5F6FA !important;
            color: #6B7280 !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 14px !important;
            padding: 13px 20px !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            width: 100% !important;
        }

        /* ── Health Score card ────────────────────────────────────── */
        .jm-score-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2px;
        }
        .jm-badge {
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.6px;
            text-transform: uppercase;
        }
        .badge-excellent { background: #E8F5E9; color: #2E7D32; }
        .badge-good      { background: #E3F2FD; color: #1565C0; }
        .badge-fair      { background: #FFF3E0; color: #E65100; }
        .badge-poor      { background: #FFEBEE; color: #C62828; }

        .jm-score-reason {
            font-size: 12px;
            color: #8A8FA8;
            margin: 0 0 12px 0;
            line-height: 1.4;
        }
        .jm-gauge-wrap {
            display: flex;
            justify-content: center;
            margin: 4px 0 12px;
        }

        /* ── Nutrient chips row ───────────────────────────────────── */
        .jm-nutrients {
            display: flex;
            justify-content: space-around;
            border-top: 1px solid #F3F4F8;
            padding-top: 16px;
            margin-top: 4px;
        }
        .jm-nutrient {
            text-align: center;
            flex: 1;
        }
        .jm-nutrient-label {
            font-size: 10px;
            font-weight: 600;
            color: #AAB0C4;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            display: block;
            margin-bottom: 5px;
        }
        .jm-nutrient-value {
            font-size: 16px;
            font-weight: 700;
            display: block;
        }
        .nv-green { color: #2E7D32; }
        .nv-amber { color: #F57F17; }
        .nv-red   { color: #C62828; }
        .nv-blue  { color: #1565C0; }
        .nv-dark  { color: #1A1A2E; }
        .jm-nutrient-sep {
            width: 1px;
            background: #F3F4F8;
            align-self: stretch;
        }

        /* ── Calorie banner ───────────────────────────────────────── */
        .jm-calorie-banner {
            display: flex;
            align-items: center;
            gap: 14px;
            background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
            border-radius: 14px;
            padding: 14px 18px;
            margin: 14px 0 0 0;
        }
        .jm-cal-icon { font-size: 26px; }
        .jm-cal-num {
            font-size: 26px;
            font-weight: 800;
            color: #2E7D32;
            line-height: 1;
        }
        .jm-cal-label {
            font-size: 12px;
            color: #558B2F;
            margin-top: 2px;
        }

        /* ── Detected foods chips ─────────────────────────────────── */
        .jm-foods-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 14px 0 0 0;
        }
        .jm-food-chip {
            background: #F0F4FF;
            border-radius: 20px;
            padding: 6px 14px;
            font-size: 13px;
            font-weight: 500;
            color: #4A4F6A;
        }

        /* ── Calorie details table ─────────────────────────────────── */
        .jm-cal-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 14px;
            font-size: 13px;
        }
        .jm-cal-table th {
            text-align: left;
            color: #AAB0C4;
            font-weight: 600;
            font-size: 11px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            padding-bottom: 8px;
            border-bottom: 1px solid #F3F4F8;
        }
        .jm-cal-table td {
            padding: 8px 0;
            color: #4A4F6A;
            border-bottom: 1px solid #F8F9FB;
        }
        .jm-cal-table td:last-child {
            text-align: right;
            font-weight: 600;
            color: #2E7D32;
        }

        /* ── Healthy Tips card ────────────────────────────────────── */
        .jm-tips-card {
            background: linear-gradient(150deg, #1B4D3E 0%, #2D6A4F 100%);
            border-radius: 20px;
            padding: 22px 22px 10px 22px;
            margin-bottom: 16px;
            box-shadow: 0 6px 20px rgba(27,77,62,0.28);
        }
        .jm-tips-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 18px;
        }
        .jm-tips-icon-wrap {
            width: 34px;
            height: 34px;
            background: rgba(255,255,255,0.15);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }
        .jm-tips-title {
            font-size: 18px;
            font-weight: 700;
            color: #ffffff;
        }
        .jm-tip-row {
            display: flex;
            align-items: flex-start;
            gap: 11px;
            margin-bottom: 14px;
        }
        .jm-tip-check {
            width: 20px;
            height: 20px;
            background: rgba(255,255,255,0.15);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            margin-top: 1px;
            font-size: 11px;
            color: #81C784;
        }
        .jm-tip-text {
            font-size: 13px;
            color: rgba(255,255,255,0.88);
            line-height: 1.55;
        }

        /* ── Error / info message ─────────────────────────────────── */
        [data-testid="stAlert"] {
            border-radius: 12px !important;
        }

        /* ── Spinner tweak ────────────────────────────────────────── */
        [data-testid="stSpinner"] > div {
            border-top-color: #4CAF50 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────── SVG gauge ───────────────────────────────

def _gauge_svg(score: int) -> str:
    r = 60
    circ = 2 * math.pi * r
    offset = circ * (1 - score / 100)
    if score >= 80:
        color = "#4CAF50"
    elif score >= 60:
        color = "#42A5F5"
    elif score >= 40:
        color = "#FFA726"
    else:
        color = "#EF5350"
    return (
        f'<svg width="160" height="160" viewBox="0 0 160 160">'
        f'<circle cx="80" cy="80" r="{r}" fill="none" stroke="#EEEFF4" stroke-width="14"/>'
        f'<circle cx="80" cy="80" r="{r}" fill="none" stroke="{color}" stroke-width="14" '
        f'stroke-linecap="round" '
        f'stroke-dasharray="{circ:.3f}" stroke-dashoffset="{offset:.3f}" '
        f'transform="rotate(-90 80 80)"/>'
        f'<text x="80" y="74" text-anchor="middle" font-size="30" font-weight="700" '
        f'fill="#1A1A2E" font-family="Inter,sans-serif">{score}</text>'
        f'<text x="80" y="94" text-anchor="middle" font-size="12" fill="#AAB0C4" '
        f'font-family="Inter,sans-serif">out of 100</text>'
        f"</svg>"
    )


def _score_badge(score: int) -> Tuple[str, str]:
    if score >= 80:
        return "EXCELLENT", "badge-excellent"
    elif score >= 60:
        return "GOOD", "badge-good"
    elif score >= 40:
        return "FAIR", "badge-fair"
    return "POOR", "badge-poor"


def _nutrient_color(level: Optional[str], kind: str) -> str:
    if not level:
        return "nv-dark"
    lv = level.lower()
    if kind == "fiber":
        return {"high": "nv-green", "medium": "nv-blue", "low": "nv-amber"}.get(lv, "nv-dark")
    if kind == "sugar":
        return {"low": "nv-green", "medium": "nv-amber", "high": "nv-red"}.get(lv, "nv-dark")
    return "nv-dark"


# ─────────────────────────── UI sections ─────────────────────────────

def render_header() -> None:
    st.markdown(
        """
        <div class="jm-header">
          <div class="jm-logo">
            <div class="jm-logo-icon">J</div>
            <span class="jm-logo-text">Jom Makan</span>
          </div>
          <div class="jm-profile-btn">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
                 viewBox="0 0 24 24" fill="none" stroke="#8A8FA8"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
              <circle cx="12" cy="7" r="4"/>
            </svg>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Dict[str, str]:
    # region agent log
    _agent_log(
        hypothesis_id="A",
        location="app.py:render_sidebar",
        message="rendering sidebar inputs",
        data={
            "env_google_api_set": bool(os.getenv("GOOGLE_API_KEY")),
            "env_google_api_len": len(os.getenv("GOOGLE_API_KEY", "")),
            "env_calorie_api_set": bool(os.getenv("USDA_CALORIE_API_KEY")),
        },
    )
    # endregion
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
            key="gemini_api_key",
            help="Your Google AI Studio API key.",
        )
        calorie_api_key = st.text_input(
            "USDA FoodData API Key (optional)",
            type="password",
            key="calorie_api_key",
            value=os.getenv("USDA_CALORIE_API_KEY", ""),
            help="Improves calorie grounding. Optional.",
        )
    # region agent log
    _agent_log(
        hypothesis_id="A",
        location="app.py:render_sidebar",
        message="sidebar captured values",
        data={
            "key_len": len(api_key),
            "calorie_key_len": len(calorie_api_key),
            "has_key": bool(api_key.strip()),
            "has_calorie_key": bool(calorie_api_key.strip()),
        },
    )
    # endregion
    return {"gemini_api_key": api_key, "calorie_api_key": calorie_api_key}


def render_upload_section(api_key: str, calorie_api_key: str) -> None:
    """Render the 'Analyze Your Meal' card."""
    st.markdown(
        """
        <div class="jm-page">
          <div class="jm-card">
            <p class="jm-card-title">Analyze Your Meal</p>
            <p class="jm-card-subtitle">
              Upload a photo of your food to get a nutrition breakdown.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div style="padding:0 16px;">', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload food photo",
            type=["png", "jpg", "jpeg", "webp", "heic"],
            accept_multiple_files=False,
            key="uploader",
            label_visibility="collapsed",
        )

        stored = st.session_state.food_image

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            mime_type = mime_from_upload(uploaded_file)
            stored = {
                "filename": uploaded_file.name,
                "mime_type": mime_type,
                "bytes": image_bytes,
            }
            st.session_state.food_image = stored

        # Preview
        if stored:
            st.image(stored["bytes"], use_container_width=True)

        # Buttons
        col_analyze, col_clear = st.columns([3, 1], gap="small")
        with col_analyze:
            analyze_clicked = st.button(
                "🔍  Analyze Meal",
                type="primary",
                disabled=stored is None,
                use_container_width=True,
                key="btn_analyze",
            )
        with col_clear:
            clear_clicked = st.button(
                "Clear",
                type="secondary",
                use_container_width=True,
                key="btn_clear",
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # region agent log
    _agent_log(
        hypothesis_id="B",
        location="app.py:render_upload_section",
        message="button interaction state",
        data={
            "api_key_has_value": bool(api_key.strip()),
            "calorie_key_has_value": bool(calorie_api_key.strip()),
            "stored_is_set": bool(stored),
            "stored_filename": stored["filename"] if stored else "",
            "analyze_clicked": "analyze_clicked" in st.session_state and st.session_state.get("analyze_clicked"),
        },
    )
    # endregion

    if clear_clicked:
        st.session_state.food_image = None
        st.session_state.analysis_result = None
        st.rerun()

    if analyze_clicked:
        # region agent log
        _agent_log(
            hypothesis_id="B",
            location="app.py:render_upload_section",
            message="analyze click processed",
            data={"api_key_has_value": bool(api_key.strip()), "stored_present": bool(stored)},
        )
        # endregion
        if not api_key.strip():
            # region agent log
            _agent_log(
                hypothesis_id="B",
                location="app.py:render_upload_section",
                message="analysis blocked due empty api key",
                data={"api_key_len": len(api_key)},
            )
            # endregion
            st.error("Please provide a Gemini API key in the ⚙️ sidebar.")
            return
        with st.spinner("Analyzing your meal…"):
            try:
                result = analyze_food_image(
                    api_key=api_key.strip(),
                    image_bytes=stored["bytes"],
                    mime_type=stored["mime_type"],
                    calorie_api_key=calorie_api_key,
                )
                st.session_state.analysis_result = result
                st.rerun()
            except Exception as err:
                # region agent log
                _agent_log(
                    hypothesis_id="D",
                    location="app.py:render_upload_section",
                    message="analysis failed",
                    data={"error_type": err.__class__.__name__, "error_message": str(err)[:200]},
                )
                # endregion
                st.error(f"Analysis failed: {err}")


def render_score_card(result: Dict[str, Any]) -> None:
    """Health Score card with circular gauge + nutrient chips."""
    score = result.get("healthy_score", 0)
    reason = result.get("reason", "")
    label, badge_cls = _score_badge(score)
    gauge = _gauge_svg(score)

    # Nutrients
    fiber = result.get("fiber_level")
    sugar = result.get("sugar_level")
    protein_g = result.get("protein_g")
    protein_display = f"{protein_g:.0f}g" if protein_g is not None else "—"
    fiber_display = fiber or "—"
    sugar_display = sugar or "—"
    fiber_cls = _nutrient_color(fiber, "fiber")
    sugar_cls = _nutrient_color(sugar, "sugar")

    # Detected foods chips
    foods = result.get("detected_foods", [])
    food_chips_html = ""
    if foods:
        chips = "".join(f'<span class="jm-food-chip">{f}</span>' for f in foods)
        food_chips_html = f'<div class="jm-foods-wrap">{chips}</div>'

    # Calorie banner
    cal = result.get("estimated_total_calories_kcal")
    cal_banner_html = ""
    if cal is not None:
        try:
            cal_val = f"{float(cal):.0f}"
        except (TypeError, ValueError):
            cal_val = str(cal)
        cal_banner_html = f"""
        <div class="jm-calorie-banner">
          <span class="jm-cal-icon">🔥</span>
          <div>
            <div class="jm-cal-num">{cal_val} kcal</div>
            <div class="jm-cal-label">Estimated total calories</div>
          </div>
        </div>"""

    # Calorie details table
    calorie_lookup = result.get("calorie_lookup", [])
    cal_table_html = ""
    if calorie_lookup:
        rows = ""
        for item in calorie_lookup:
            name = item.get("matched_name") or item.get("food_name") or "Unknown"
            kcal = item.get("calories_kcal")
            serving = item.get("serving_grams")
            kcal_str = f"{float(kcal):.0f} kcal" if kcal is not None else "—"
            srv_str = f" / {serving:.0f}g" if serving else ""
            rows += f"<tr><td>{name}{srv_str}</td><td>{kcal_str}</td></tr>"
        cal_table_html = f"""
        <table class="jm-cal-table">
          <thead><tr><th>Food item</th><th style="text-align:right;">Calories</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>"""

    st.markdown(
        f"""
        <div class="jm-page">
          <div class="jm-card">
            <div class="jm-score-row">
              <p class="jm-card-title" style="margin:0;">Health Score</p>
              <span class="jm-badge {badge_cls}">{label}</span>
            </div>
            <p class="jm-score-reason">{reason}</p>
            <div class="jm-gauge-wrap">{gauge}</div>
            <div class="jm-nutrients">
              <div class="jm-nutrient">
                <span class="jm-nutrient-label">Fiber</span>
                <span class="jm-nutrient-value {fiber_cls}">{fiber_display}</span>
              </div>
              <div class="jm-nutrient-sep"></div>
              <div class="jm-nutrient">
                <span class="jm-nutrient-label">Sugar</span>
                <span class="jm-nutrient-value {sugar_cls}">{sugar_display}</span>
              </div>
              <div class="jm-nutrient-sep"></div>
              <div class="jm-nutrient">
                <span class="jm-nutrient-label">Protein</span>
                <span class="jm-nutrient-value nv-dark">{protein_display}</span>
              </div>
            </div>
            {food_chips_html}
            {cal_banner_html}
            {cal_table_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tips_card(result: Dict[str, Any]) -> None:
    """Healthy Tips card with dark-green gradient background."""
    tips = result.get("improvement_tips", [])
    if not tips:
        return

    tip_rows = ""
    for tip in tips:
        tip_rows += f"""
        <div class="jm-tip-row">
          <div class="jm-tip-check">✓</div>
          <span class="jm-tip-text">{tip}</span>
        </div>"""

    st.markdown(
        f"""
        <div class="jm-page">
          <div class="jm-tips-card">
            <div class="jm-tips-header">
              <div class="jm-tips-icon-wrap">⚡</div>
              <span class="jm-tips-title">Healthy Tips</span>
            </div>
            {tip_rows}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────── entry point ─────────────────────────────

def main() -> None:
    _agent_log(
        hypothesis_id="A",
        location="app.py:main",
        message="app start",
        data={
            "env_google_api_len": len(os.getenv("GOOGLE_API_KEY", "")),
            "env_usda_api_len": len(os.getenv("USDA_CALORIE_API_KEY", "")),
            "working_dir": os.getcwd(),
        },
    )
    load_dotenv(override=False)
    st.set_page_config(
        page_title="Jom Makan",
        page_icon="🍽️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    init_session_state()
    inject_css()
    config = render_sidebar()

    render_header()
    render_upload_section(
        api_key=config["gemini_api_key"],
        calorie_api_key=config["calorie_api_key"],
    )

    result = st.session_state.analysis_result
    if result:
        render_score_card(result)
        render_tips_card(result)


if __name__ == "__main__":
    main()
