# PCOS & Endometriosis Differential Diagnostician

A three-layer clinical decision support tool that helps distinguish polycystic ovary syndrome (PCOS) from endometriosis — two conditions with heavily overlapping symptoms (pelvic pain, irregular cycles, hormonal imbalance, infertility) that are frequently confused, contributing to diagnostic delays of 7–10 years.

Built for **BioHackathon 2026** (NTU Biological Sciences Club) — **Honourable Mention**, out of approximately 30 teams.

> **For education and decision support only. Not a medical diagnosis and not a certified medical device.**

---

## My contribution

This was a four-person hackathon project. I built the application layer, the explainability work, the deployment, and the pitch; the underlying classifiers were trained by teammates. Specifically:

### The application (`app.py`)

The full Streamlit interface implementing the three-layer diagnostic pathway — a staged flow where each layer runs only if the previous one was inconclusive, mirroring how a clinician actually works through a differential diagnosis.

**Handling incomplete patient data.** The PCOS model needs 41 features. No clinician has 41 features to hand. I designed a simple form asking for the 14 that matter most and filling the remainder with training-set medians, so the tool returns something useful on partial data rather than refusing to run. The complete feature vector actually sent to the model stays viewable in the UI, so nothing is hidden from the user.

**Input validation.** Values outside the training range raise an error; values in the unusual tails raise a warning. A typo or an out-of-distribution patient becomes visible instead of silently producing a confident-looking prediction — which, in a clinical context, is the failure mode that actually matters.

**Two input modes.** A simple form for the fields that carry most of the signal, and an advanced expander exposing all 41 PCOS features and the full endometriosis input set for users who want full control.

**Cross-model field syncing.** BMI is derived from height and weight rather than asked for separately, and age and BMI carry across from the PCOS form into the endometriosis model so the same patient isn't entered twice.

**Layer 3 rule-out logic.** Threshold checks for conditions that mimic both — TSH > 4.5 or < 0.4 mIU/L for thyroid dysfunction, prolactin > 25 ng/mL for hyperprolactinemia — so an inconclusive result still produces an actionable next step rather than a shrug.

### Explainability

I ran SHAP analysis (`TreeExplainer`) over the trained Random Forest models to surface per-feature contributions. A model that flags a patient without saying which markers drove the decision isn't something a clinician can reasonably act on. The analysis identified follicle count (left and right), AMH and the hyperandrogenic markers — weight gain, skin darkening, hair growth — as the dominant drivers. Surfacing patient-level SHAP inside the UI is on the roadmap below; the app currently exposes global feature importance.

### Deployment

Models are persisted with `joblib` and loaded through `@st.cache_resource`, so the app starts instantly rather than retraining on every launch. For judging I exposed it over public HTTPS via localtunnel, so the panel could open the tool in a browser during our slot and use it themselves instead of watching a recorded demo. It runs from a single command with no GPU and no cloud account.

### Presentation

I wrote and delivered the pitch to the judging panel — model performance, three worked patient cases walking through each layer of the pathway, an honest account of where the endometriosis model falls short, and a three-phase implementation roadmap.

---

## How the pathway works

**Layer 1 — PCOS classifier.** Random Forest (100 trees, balanced class weighting) over 41 clinical features spanning demographics, hormone panels, ultrasound findings, symptoms and lifestyle, standardised with `StandardScaler`. If P(PCOS) > 0.60 the pathway stops and reports PCOS as likely.

**Layer 2 — Endometriosis classifier.** Random Forest over 6 features (chronic pain level, BMI, age, menstrual irregularity, hormone abnormality, infertility). Runs only when Layer 1 falls below threshold.

**Layer 3 — Clinical rule-outs.** Threshold logic catching conditions that mimic both, otherwise recommending referral to reproductive endocrinology.

## Results

| | PCOS model | Endometriosis model |
|---|---|---|
| AUC-ROC | 0.962 | 0.603 |
| Accuracy | 89.6% | 59.9% |
| F1 | 0.825 | 0.453 |
| Patients | 541 | 10,000 |
| Features | 41 | 6 |

Evaluated with 5-fold stratified cross-validation; class imbalance in the PCOS set (177 positive / 364 negative) handled with balanced class weighting.

**On the endometriosis model:** its performance is poor, and we said so in the pitch rather than hiding it. It trains on synthetic data with only 6 features, which is not enough signal for a condition normally diagnosed by laparoscopy. Layer 2 should be read as a flag for further workup, not a prediction. Improving it is the first item below.

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/marvintzr/biohackathon-pcos-endometriosis.git
cd biohackathon-pcos-endometriosis
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. On Windows, `run.bat` does the same in one double-click.

### Datasets

**Not included in this repository** — they were provided by the hackathon organisers and are not mine to redistribute. To run the app, place both files in a `data/` folder inside the project:

- `(Main_Dataset)_PCOS_data_without_infertility.xlsx` — PCOS clinical parameters, 541 patients (publicly available Kerala fertility-clinic dataset)
- `(Supplementary_Dataset)_structured_endometriosis_data.xlsx` — synthetic endometriosis features, 10,000 records

### Models

Also not committed — large binaries, reproducible from the data. The app trains both on first run (1–2 minutes) and caches them to `models/`. There's a "Retrain models & refresh stats" button in the sidebar.

## Project structure

```
app.py             Streamlit UI, three-layer pathway, validation, result display
ml_pipeline.py     Data loading, imputation, model training, persistence, inference
feature_stats.py   Per-column statistics, field grouping, range and outlier checks
train.py           Standalone training entry point
requirements.txt   Dependencies
run.bat            One-click Windows launcher
```

## Roadmap

- Patient-level SHAP waterfall plots surfaced directly in the UI
- Real EHR endometriosis data to replace the synthetic Layer 2 training set
- Unit tests and a CI pipeline
- Docker containerisation
- PCOS subtype classification (lean / obese / classic)
- HL7/FHIR integration for use alongside existing hospital systems

## Licence

MIT
