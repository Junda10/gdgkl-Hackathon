import json
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

VISION_ANALYSIS_PROMPT = (
    "You are a nutrition assistant. Analyze the uploaded food image and return ONLY "
    "valid JSON with keys:\n"
    "detected_foods (array of up to 5 likely food items),\n"
    "healthy_score (integer 0-100),\n"
    "reason (short summary),\n"
    "improvement_tips (array of up to 5 short tips).\n"
    "Respond with concise, realistic health guidance and no markdown."
)

CALORIE_REFINEMENT_PROMPT = (
    "You are a nutrition assistant. You have image analysis results and calorie "
    "lookup data from a food-nutrition API. Return ONLY JSON with keys:\n"
    "detected_foods (array of food names),\n"
    "estimated_total_calories_kcal (number),\n"
    "healthy_score (integer 0-100),\n"
    "reason (short summary),\n"
    "improvement_tips (array of up to 5 short tips).\n"
    "No markdown."
)


def init_session_state() -> None:
    """Initialize image storage and analysis result fields."""
    if "food_image" not in st.session_state:
        st.session_state.food_image = None
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None


def mime_from_upload(uploaded_file) -> str:
    """Return a MIME type accepted by Gemini from upload metadata."""
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
    """Parse strict JSON from a model response that may include extra text."""
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
    """Validate/normalize model output into a stable structure."""
    score_raw = raw.get("healthy_score")
    tips_raw = raw.get("improvement_tips")
    reason = raw.get("reason", "No reason provided.")
    foods_raw = raw.get("detected_foods", raw.get("foods", []))

    try:
        score = int(score_raw)
    except (TypeError, ValueError):
        return {}, f"Could not parse healthy_score: {score_raw!r}"

    detected_foods = foods_raw if isinstance(foods_raw, list) else [str(foods_raw)] if foods_raw else []
    detected_foods = [str(food).strip() for food in detected_foods if str(food).strip()]

    score = max(0, min(100, score))
    if not isinstance(tips_raw, list):
        tips_raw = [str(tips_raw)] if tips_raw else []

    tips = [str(tip).strip() for tip in tips_raw if str(tip).strip()]
    if not tips:
        tips = ["Improve nutrition quality in future analysis."]

    estimated_calories_raw = raw.get(
        "estimated_total_calories_kcal",
        raw.get("total_estimated_calories_kcal", raw.get("total_calories_kcal")),
    )
    try:
        estimated_calories_kcal = float(estimated_calories_raw) if estimated_calories_raw is not None else None
    except (TypeError, ValueError):
        estimated_calories_kcal = None

    calorie_lookup = raw.get("calorie_lookup", raw.get("calorie_data", []))
    if not isinstance(calorie_lookup, list):
        calorie_lookup = [calorie_lookup] if calorie_lookup else []

    normalized_calories = []
    for item in calorie_lookup:
        if not isinstance(item, dict):
            continue
        normalized_item = {
            "food_name": str(item.get("food_name", item.get("name", ""))).strip(),
            "calories_kcal": item.get("calories_kcal"),
            "matched_name": str(item.get("matched_name", "")).strip(),
            "serving_grams": item.get("serving_grams"),
            "source": str(item.get("source", "calorie_api")).strip() or "calorie_api",
        }
        if normalized_item["food_name"] and normalized_item["calories_kcal"] is not None:
            normalized_calories.append(normalized_item)

    return {
        "healthy_score": score,
        "reason": str(reason).strip() or "No reason provided.",
        "improvement_tips": tips[:5],
        "detected_foods": detected_foods[:10],
        "estimated_total_calories_kcal": estimated_calories_kcal,
        "calorie_lookup": normalized_calories,
    }, None


def fetch_usda_calories(food_name: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Query USDA FoodData Central for an estimated calorie value of one food."""
    params = {
        "query": food_name,
        "pageSize": 1,
        "api_key": api_key,
    }
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
            raw_value = nutrient.get("value")
            try:
                calories = float(raw_value)
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
    foods: List[str],
    api_key: str,
) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """Fetch calories per item and aggregate estimated total."""
    lookup = []
    total = 0.0
    has_data = False

    for food in foods:
        food_name = str(food).strip()
        if not food_name:
            continue

        try:
            item = fetch_usda_calories(food_name, api_key)
        except requests.RequestException:
            continue
        except ValueError:
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
    """Re-run analysis to make score and tips use structured calorie context."""
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

    parsed = extract_json(text.strip())
    return parsed


def analyze_food_image(
    api_key: str,
    image_bytes: bytes,
    mime_type: str,
    calorie_api_key: str = "",
) -> Dict[str, Any]:
    """Call Gemini Vision and return a normalized analysis payload."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            VISION_ANALYSIS_PROMPT,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
    )
    text = getattr(response, "text", "")
    if text is None:
        reason = ""
        try:
            if response.candidates:
                reason = str(response.candidates[0].finish_reason)
        except Exception:
            reason = "unknown"
        raise RuntimeError(
            "Gemini did not return analysis text. "
            f"Finish reason: {reason or 'unknown'}."
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
        foods=detected_foods,
        api_key=calorie_api_key.strip(),
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


def load_api_keys() -> Dict[str, str]:
    return {
        "gemini_api_key": os.getenv("GOOGLE_API_KEY", ""),
        "calorie_api_key": os.getenv("USDA_CALORIE_API_KEY", ""),
    }


def render_uploader() -> Optional[dict]:
    """Render upload controls and persist uploaded image in session state."""
    uploaded_file = st.file_uploader(
        "Upload a food photo",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        accept_multiple_files=False,
        help="Upload one clear food image for analysis.",
        key="uploader",
    )

    if uploaded_file is None:
        return None

    image_bytes = uploaded_file.getvalue()
    mime_type = mime_from_upload(uploaded_file)
    st.session_state.food_image = {
        "filename": uploaded_file.name,
        "mime_type": mime_type,
        "bytes": image_bytes,
    }
    return st.session_state.food_image


def render_preview(stored_image: Optional[dict]) -> None:
    if not stored_image:
        st.info("Upload an image to see a preview and start analysis.")
        return

    st.image(
        stored_image["bytes"],
        caption=stored_image["filename"],
        use_container_width=True,
    )


def render_analysis_button(stored_image: Optional[dict], api_key: str) -> None:
    if not stored_image:
        st.button("Analyze food", type="primary", disabled=True)
        return

    if st.button("Analyze food", type="primary"):
        if not api_key.strip():
            st.error("Please set GOOGLE_API_KEY in your .env file.")
            return

        with st.spinner("Analyzing your food image..."):
            try:
                result = analyze_food_image(
                    api_key=api_key.strip(),
                    image_bytes=stored_image["bytes"],
                    mime_type=stored_image["mime_type"],
                    calorie_api_key=st.session_state.get("calorie_api_key", ""),
                )
                st.session_state.analysis_result = result
                st.success("Analysis complete!")
            except Exception as err:
                st.error(f"Analysis failed: {err}")


def render_results() -> None:
    result = st.session_state.analysis_result
    if not result:
        return

    st.divider()
    st.header("Healthy score")
    score = result.get("healthy_score", 0)
    st.metric("Healthy Score", f"{score} / 100")
    st.write(result.get("reason", ""))

    detected_foods = result.get("detected_foods", [])
    if detected_foods:
        st.subheader("Detected foods")
        for idx, food in enumerate(detected_foods, start=1):
            st.write(f"{idx}. {food}")

    estimated_calories = result.get("estimated_total_calories_kcal")
    if estimated_calories is not None:
        try:
            calorie_display = f"{float(estimated_calories):.0f} kcal"
        except (TypeError, ValueError):
            calorie_display = str(estimated_calories)
        st.metric("Estimated calories", calorie_display)

    calorie_lookup = result.get("calorie_lookup", [])
    if calorie_lookup:
        st.subheader("Calorie lookup details")
        for item in calorie_lookup:
            food_name = item.get("food_name", item.get("matched_name", "")) or "Unknown"
            cal = item.get("calories_kcal")
            matched = item.get("matched_name") or food_name
            if cal is None:
                st.write(f"- {food_name}: no calorie match")
                continue
            serving = item.get("serving_grams")
            serving_text = f" (~{serving}g)" if serving else ""
            st.write(f"- {matched}{serving_text}: {float(cal):.0f} kcal ({item.get('source', 'calorie_api')})")

    st.subheader("Tips to improve")
    for idx, tip in enumerate(result.get("improvement_tips", []), start=1):
        st.write(f"{idx}. {tip}")


def render_clear_action() -> None:
    if st.button("Clear uploaded image"):
        st.session_state.food_image = None
        st.session_state.analysis_result = None


def main() -> None:
    load_dotenv(override=False)
    st.set_page_config(page_title="JomMakan", page_icon="🍽️", layout="centered")
    st.title("JomMakan")
    st.write("Upload food photos and get a healthy score plus practical improvement tips.")

    init_session_state()
    config = load_api_keys()
    stored_image = render_uploader()
    render_preview(stored_image)
    render_analysis_button(stored_image, config["gemini_api_key"])
    render_clear_action()
    render_results()


if __name__ == "__main__":
    main()
