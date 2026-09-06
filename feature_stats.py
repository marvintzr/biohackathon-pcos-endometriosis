"""Column statistics from training data for UI bounds and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# UI grouping for all 41 PCOS features (excluding target).
PCOS_FIELD_GROUPS: dict[str, list[str]] = {
    "Demographics": [
        "Age (yrs)",
        "Weight (Kg)",
        "Height(Cm)",
        "BMI",
        "Blood Group",
    ],
    "Vitals": [
        "Pulse rate(bpm)",
        "RR (breaths/min)",
        "Hb(g/dl)",
        "BP _Systolic (mmHg)",
        "BP _Diastolic (mmHg)",
    ],
    "Menstrual & reproductive": [
        "Cycle(R/I)",
        "Cycle length(days)",
        "Marraige Status (Yrs)",
        "Pregnant(Y/N)",
        "No. of abortions",
        "I   beta-HCG(mIU/mL)",
        "II    beta-HCG(mIU/mL)",
    ],
    "Hormones & labs": [
        "FSH(mIU/mL)",
        "LH(mIU/mL)",
        "FSH/LH",
        "TSH (mIU/L)",
        "AMH(ng/mL)",
        "PRL(ng/mL)",
        "Vit D3 (ng/mL)",
        "PRG(ng/mL)",
        "RBS(mg/dl)",
    ],
    "Symptoms & lifestyle": [
        "Weight gain(Y/N)",
        "hair growth(Y/N)",
        "Skin darkening (Y/N)",
        "Hair loss(Y/N)",
        "Pimples(Y/N)",
        "Fast food (Y/N)",
        "Reg.Exercise(Y/N)",
    ],
    "Body composition": [
        "Hip(inch)",
        "Waist(inch)",
        "Waist:Hip Ratio",
    ],
    "Ultrasound": [
        "Follicle No. (L)",
        "Follicle No. (R)",
        "Avg. F size (L) (mm)",
        "Avg. F size (R) (mm)",
        "Endometrium (mm)",
    ],
}

# Shown in the default (simple) Streamlit UI — aligned with Rotterdam / notebook story.
SIMPLE_PCOS_FIELDS: list[str] = [
    "Age (yrs)",
    "Weight (Kg)",
    "Height(Cm)",
    "Cycle length(days)",
    "hair growth(Y/N)",
    "Weight gain(Y/N)",
    "Skin darkening (Y/N)",
    "Fast food (Y/N)",
    "Follicle No. (R)",
    "Follicle No. (L)",
    "AMH(ng/mL)",
    "PRL(ng/mL)",
    "TSH (mIU/L)",
]

# Layer 2: pain is the main extra input; age/BMI sync from PCOS simple form.
SIMPLE_ENDO_FIELDS: list[str] = [
    "Chronic_Pain_Level",
]

ENDO_FIELD_GROUPS: dict[str, list[str]] = {
    "Endometriosis model inputs": [
        "Age",
        "Menstrual_Irregularity",
        "Chronic_Pain_Level",
        "Hormone_Level_Abnormality",
        "Infertility",
        "BMI",
    ],
}


@dataclass
class ColumnStats:
    name: str
    min_val: float
    max_val: float
    median: float
    p05: float
    p95: float
    n_unique: int
    choices: list[float]  # sorted unique values (for discrete / categorical)
    kind: str  # "binary", "discrete", "continuous"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "median": self.median,
            "p05": self.p05,
            "p95": self.p95,
            "n_unique": self.n_unique,
            "choices": self.choices,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ColumnStats:
        return cls(**d)


def _series_stats(name: str, series: pd.Series) -> ColumnStats:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        s = pd.Series([0.0])
    choices = sorted(s.unique().tolist())
    n_unique = len(choices)
    if n_unique <= 2:
        kind = "binary"
    elif n_unique <= 12:
        kind = "discrete"
    else:
        kind = "continuous"
    return ColumnStats(
        name=name,
        min_val=float(s.min()),
        max_val=float(s.max()),
        median=float(s.median()),
        p05=float(np.percentile(s, 5)),
        p95=float(np.percentile(s, 95)),
        n_unique=n_unique,
        choices=choices,
        kind=kind,
    )


def compute_dataset_stats(
    df: pd.DataFrame, target_col: str | None = None
) -> dict[str, ColumnStats]:
    cols = [c for c in df.columns if c != target_col]
    return {col: _series_stats(col, df[col]) for col in cols}


def median_patient_row(stats: dict[str, ColumnStats], columns: list[str]) -> dict[str, float]:
    return {col: stats[col].median if col in stats else 0.0 for col in columns}


def merge_patient_row(
    columns: list[str],
    stats: dict[str, ColumnStats],
    overrides: dict[str, float],
) -> pd.DataFrame:
    row = median_patient_row(stats, columns)
    row.update(overrides)
    return pd.DataFrame([row], columns=columns)


def out_of_range_warnings(
    values: dict[str, float], stats: dict[str, ColumnStats]
) -> list[str]:
    warnings: list[str] = []
    for col, val in values.items():
        if col not in stats:
            continue
        st = stats[col]
        if val < st.min_val or val > st.max_val:
            warnings.append(
                f"**{col}**: {val:g} is outside training range "
                f"[{st.min_val:g}, {st.max_val:g}] (median {st.median:g})."
            )
    return warnings


def unusual_value_warnings(
    values: dict[str, float], stats: dict[str, ColumnStats]
) -> list[str]:
    """Flag values outside 5th–95th percentile (still in min–max but uncommon)."""
    notes: list[str] = []
    for col, val in values.items():
        if col not in stats:
            continue
        st = stats[col]
        if st.kind == "continuous" and (val < st.p05 or val > st.p95):
            if st.min_val <= val <= st.max_val:
                notes.append(
                    f"**{col}**: {val:g} is uncommon in training "
                    f"(typical range {st.p05:g}–{st.p95:g})."
                )
    return notes
