# JomMakan

JomMakan is a simple Streamlit app that lets users upload a food photo, then uses
Google Gemini Vision to generate a healthy score and improvement tips.

## Requirements

- Python 3.10+
- Google AI Studio API key with access to Gemini (used as `GOOGLE_API_KEY`)
- Optional USDA FoodData Central API key for calorie lookup (`USDA_CALORIE_API_KEY`)
- Internet connection for Gemini API calls

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   ```
2. Activate it:
   - Windows (PowerShell):
     ```bash
     .\.venv\Scripts\Activate.ps1
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment template:
   ```bash
   copy .env.example .env
   ```
5. Add your Google AI Studio key (required) and optional USDA key in `.env`.
   - USDA key: https://fdc.nal.usda.gov/api-key-signup

## Run

```bash
streamlit run app.py
```

## Notes

- Calorie API is optional. When provided, JomMakan queries USDA FoodData Central for top calorie matches and sends matched values back to Gemini for refined scoring.
