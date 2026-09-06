# PCOS & Endometriosis Differential Diagnostician

A three-layer clinical decision support tool that helps distinguish polycystic ovary syndrome (PCOS) from endometriosis — two conditions with heavily overlapping symptoms (pelvic pain, irregular cycles, hormonal imbalance, infertility) that are frequently confused, contributing to diagnostic delays of 7–10 years.

Built for **BioHackathon 2026** (NTU Biological Sciences Club), where it was awarded **Honourable Mention** out of approximately 30 teams.

> **For education and decision support only. Not a medical diagnosis and not a certified medical device.**

---

## How it works

The app mirrors how a clinician reasons through a differential diagnosis: rule in the most likely condition first, then the next, then check for conditions that mimic both. Each layer runs only if the previous one was inconclusive.

**Layer 1 — PCOS classifier**
Random Forest (100 trees, `class_weight="balanced"`) over 41 clinical features covering demographics, hormone panels, ultrasound findings, symptoms and lifestyle. Features are standardised with `StandardScaler`. If P(PCOS) > 0.60, the pathway stops and reports PCOS as likely.

**Layer 2 — Endometriosis classifier**
Random Forest over 6 clinical features (chronic pain level, BMI, age, menstrual irregularity, hormone abnormality, infertility). Runs only when Layer 1 falls below threshold. If P(endometriosis) > 0.60, the pathway reports endometriosis as possible and recommends further workup.

**Layer 3 — Clinical rule-outs**
Threshold logic for conditions that mimic both. Flags TSH > 4.5 or < 0.4 mIU/L (thyroid dysfunction) and prolactin > 25 ng/mL (hyperprolactinemia), and otherwise recommends referral to reproductive endocrinology.

## Design decisions worth noting

**Medians for unobserved features.** The PCOS model needs 41 features, but no clinician has all 41 to hand. The simple form asks for 14 key inputs and fills the rest with training-set medians, so the tool returns something useful on partial data rather than refusing to run. The full feature vector actually sent to the model is viewable in the UI.

**Input validation.** Values outside the training range raise an error, and values in the unusual tails raise a warning — so a typo or an out-of-distribution patient is visible rather than silently producing a confident-looking prediction.

**Two input modes.** A simple form for the fields that matter most, and an advanced expander exposing all 41 PCOS features and the full endometriosis input set.

**BMI is derived** from height and weight rather than asked for separately, and age and BMI carry across from the PCOS form into the endometriosis model.

## Results

| | PCOS model | Endometriosis model |
|---|---|---|
| AUC-ROC | 0.962 | 0.603 |
| Accuracy | 89.6% | 59.9% |
| F1 | 0.825 | 0.453 |
| Patients | 541 | 10,000 |
| Features | 41 | 6 |

Evaluated with 5-fold stratified cross-validation. Class imbalance (177 positive / 364 negative in the PCOS set) was handled with balanced class weighting.

**On the endometriosis model:** its performance is weak, and that is stated plainly rather than hidden. It is trained on a synthetic dataset with only 6 features, which is not enough signal for a condition normally diagnosed by laparoscopy. Layer 2 should be read as a flag for further workup, not a prediction. Improving it with real EHR data and richer features is the first item on the roadmap.

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/marvintzr/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

On Windows, `run.bat` does the same thing in one double-click, creating a virtual environment on first run.

### Datasets

**The datasets are not included in this repository.** They were provided by the BioHackathon organisers and are not mine to redistribute. To run the app, place both files in a `data/` folder inside the project:

- `(Main_Dataset)_PCOS_data_without_infertility.xlsx` — PCOS clinical parameters, 541 patients
- `(Supplementary_Dataset)_structured_endometriosis_data.xlsx` — synthetic endometriosis features, 10,000 records

The PCOS dataset is the publicly available Kerala fertility-clinic dataset. The endometriosis dataset was synthetic data supplied for the hackathon.

### Models

Trained models are not committed either — they are large binaries and are reproducible from the data. On first run the app trains both models automatically (about 1–2 minutes) and caches them to `models/`. The sidebar has a "Retrain models & refresh stats" button.

## Project structure

```
app.py             Streamlit UI, the three-layer pathway, validation and result display
ml_pipeline.py     Data loading, imputation, model training, persistence, inference
feature_stats.py   Per-column statistics, field grouping, out-of-range and unusual-value checks
train.py           Standalone training entry point
requirements.txt   Dependencies
run.bat            One-click Windows launcher
```

## Roadmap

- Patient-level SHAP waterfall plots surfaced in the UI (SHAP analysis was done post-training with `TreeExplainer`; the app currently shows global feature importance)
- Real EHR endometriosis data to replace the synthetic Layer 2 training set
- Unit tests and a CI pipeline
- Docker containerisation
- PCOS subtype classification (lean / obese / classic)
- HL7/FHIR integration for use alongside existing hospital systems

## Team

Built by a team of four at BioHackathon 2026.

**Tan Zhi Rong Marvin** — Streamlit application, SHAP explainability analysis, deployment, and the pitch presentation.

*(Teammates: add names here.)*

Model training and the underlying classifiers were built by teammates.

## Licence

MIT — see `LICENSE`.
