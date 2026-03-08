# JomMakan

JomMakan is a Streamlit app that lets users upload a food photo, then uses Google Gemini Vision to generate a health score, macro breakdown, calorie estimate, and personalised improvement tips.

---

## Inspiration

Malaysia's food culture is rich, vibrant, and deeply social — "Jom Makan" literally means *"Let's eat!"* in Malay. But while we love our nasi lemak, char kway teow, and mamak roti canai, it can be hard to know how nutritious our meals really are. We were inspired to build a tool that makes nutrition awareness effortless: just snap a photo and get instant, AI-powered feedback — no calorie counting apps, no manual food logging.

## What it does

JomMakan lets you upload a photo of any meal and instantly receive:
- 🏅 A **Health Score** (0–100) with a letter grade (A+ to D)
- 📊 **Macro breakdowns** — Fiber, Sugar, and Protein levels visualised as animated progress bars
- 🔥 **Estimated calorie count**, grounded with real data from the USDA FoodData Central database
- 🍜 **Detected food items** displayed as chips
- ⚡ **Personalised healthy tips** to improve the nutritional value of your meal
- 🎉 A confetti burst for excellent scores (80+)

## How we built it

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/) with a fully custom mobile-first CSS UI (glassmorphism header, animated SVG gauge, shimmer skeleton loader, confetti)
- **AI Vision Model:** Google Gemini 2.5 Flash (`gemini-2.5-flash`) via the `google-genai` SDK — used both for food image analysis and for calorie-refined re-scoring
- **Calorie Grounding:** [USDA FoodData Central API](https://fdc.nal.usda.gov/) to look up per-food calorie data, which is fed back into Gemini for a refined analysis pass
- **Containerisation:** Docker with a minimal Python 3.12-slim image
- **Deployment:** Google Cloud Run (Asia Southeast 1), with secrets managed via Google Secret Manager
- **CI/CD:** GitHub Actions with Workload Identity Federation (keyless OIDC authentication) for zero-credential automated deploys

## Challenges we ran into

- **Reliable JSON extraction from Gemini:** The model occasionally returned markdown-wrapped or partially malformed JSON. We built a robust extraction and normalisation pipeline (`extract_json` + `normalize_result`) to handle all edge cases gracefully.
- **Two-pass calorie refinement:** Orchestrating the flow of image analysis → USDA lookup → Gemini re-score required careful state management and fallback logic so a failed second pass never breaks the first result.
- **Mobile-first Streamlit UI:** Streamlit is built for data dashboards, not polished consumer apps. Achieving a native-app feel (sticky header, full-width cards, animated gauges, shimmer skeletons) required extensive CSS injection and careful layout overrides.
- **Keyless Cloud Run deployment:** Setting up Workload Identity Federation between GitHub Actions and Google Cloud for the first time had a steep learning curve, but results in a fully secure, key-free CI/CD pipeline.

## Accomplishments that we're proud of

- A **production-ready, publicly accessible app** deployed on Google Cloud Run at [jommakan-171023813626.asia-southeast1.run.app](https://jommakan-171023813626.asia-southeast1.run.app/)
- A genuinely **beautiful, mobile-first UI** built entirely within Streamlit's constraints — animated SVG donut gauge, shimmer loading skeleton, confetti celebration, and smooth card entrance animations
- A **two-stage AI pipeline** that combines Gemini's vision capabilities with real-world USDA nutrition data for more accurate calorie estimates
- A **fully automated CI/CD pipeline** using GitHub Actions and Google Cloud Workload Identity Federation — no long-lived service account keys anywhere

## What we learned

- How to use **Gemini's multimodal vision API** to extract structured nutritional data from food images
- How to build a **multi-step LLM pipeline** that enriches AI output with external API data (USDA) and feeds it back for a refined second pass
- Practical techniques for **customising Streamlit's UI** far beyond its default components using injected CSS and raw HTML
- How to set up **keyless, OIDC-based authentication** between GitHub Actions and Google Cloud Run for secure automated deployments

## What's next for JomMakan

- 📸 **Camera capture** — allow users to snap a photo directly in the app on mobile without needing to upload a file
- 🗓️ **Meal history & daily tracking** — persist results across sessions so users can monitor their nutritional habits over time
- 🇲🇾 **Local food database** — supplement USDA data with a Malaysia-specific food database for more accurate calorie estimates on local dishes
- 🤝 **Social sharing** — let users share their health scores with friends to encourage healthy eating together
- 🔔 **Personalised goals** — set daily calorie or macro targets and track progress towards them

---

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

## Local Docker run

```bash
docker build -t jommakan .
docker run -p 8080:8080 \
  -e GOOGLE_API_KEY=your_google_api_key \
  -e USDA_CALORIE_API_KEY=your_usda_key_if_available \
  jommakan
```

Then open http://localhost:8080 in your browser.

## Deploy to Google Cloud Run

This project includes:
- `Dockerfile` for containerizing the Streamlit app.
- `.dockerignore` for clean image builds.
- `.github/workflows/deploy-cloudrun.yml` for automated deployment.

### Prerequisites
- A Google Cloud project with billing enabled.
- `gcloud` CLI configured locally.
- Secret Manager secrets:
  - `google-api-key` (required): contains your Gemini API key.
  - `usda-calorie-api-key` (optional): contains your USDA key.
- A workload identity provider and service account configured for GitHub Actions.

### GitHub repository variables
Configure these GitHub variables in your repo:
- `GCP_PROJECT_ID`
- `GCP_REGION` (for example: `us-central1`)
- `GAR_REPOSITORY` (for example: `jom-repo`)
- `CLOUD_RUN_SERVICE` (for example: `jommakan`)
- `WIF_PROVIDER` (full resource name of the workload identity provider)
- `WIF_SERVICE_ACCOUNT` (service account email used for deployment)

### Deploy
1. Push to `main` to trigger the workflow automatically, or run it manually from GitHub Actions.
2. Monitor logs in the Actions tab and open the Cloud Run service URL printed by the workflow.

### Notes
- The workflow requires `google-api-key`; `usda-calorie-api-key` is optional and skipped automatically if missing.
- The container exposes port `8080` and runs:
  `streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0`.
- First-time deployment should be authenticated via GitHub Actions OIDC; no long-lived service account keys are needed.

## Notes

- Calorie API is optional. When provided, JomMakan queries USDA FoodData Central for top calorie matches and sends matched values back to Gemini for refined scoring.

https://jommakan-171023813626.asia-southeast1.run.app/

Video demo link
https://drive.google.com/file/d/1HSn6CkM3cwcG3r9q-YkMBwvKKfNaiD6h/view?usp=sharing