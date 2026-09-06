"""Train and load PCOS / endometriosis Random Forest models (notebook-aligned)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODELS_DIR = APP_DIR / "models"

PCOS_FILE = "(Main_Dataset)_PCOS_data_without_infertility.xlsx"
ENDO_FILE = "(Supplementary_Dataset)_structured_endometriosis_data.xlsx"

PCOS_COERCE = [
    "AMH(ng/mL)",
    "II    beta-HCG(mIU/mL)",
    "Marraige Status (Yrs)",
    "Fast food (Y/N)",
]


def _has_datasets(folder: Path) -> bool:
    return (folder / PCOS_FILE).is_file() and (folder / ENDO_FILE).is_file()


def _resolve_data_dir() -> Path:
    """Look only inside this project folder (portable zip / shared copy)."""
    candidates = [
        DATA_DIR,
        APP_DIR,
    ]
    for folder in candidates:
        if _has_datasets(folder):
            return folder.resolve()
    return DATA_DIR.resolve()


def expected_data_paths() -> str:
    """Human-readable paths for error messages."""
    return (
        f"  - {DATA_DIR / PCOS_FILE}\n"
        f"  - {DATA_DIR / ENDO_FILE}\n"
        f"  (or the same two filenames in {APP_DIR})"
    )


def require_data_dir() -> Path:
    root = _resolve_data_dir()
    if not _has_datasets(root):
        raise FileNotFoundError(
            "Dataset files not found inside the app folder.\n"
            "Expected:\n"
            f"{expected_data_paths()}"
        )
    return root


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].median())
        else:
            out[col] = out[col].fillna(out[col].mode().iloc[0])
    return out


def load_pcos_dataframe(data_dir: Path | None = None) -> pd.DataFrame:
    root = data_dir or require_data_dir()
    path = root / PCOS_FILE
    if not path.exists():
        raise FileNotFoundError(f"PCOS dataset not found: {path}\n{expected_data_paths()}")

    df = pd.read_excel(path, sheet_name=1)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["Sl. No", "Patient File No."], errors="ignore")
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    for col in PCOS_COERCE:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return _impute(df)


def load_endo_dataframe(data_dir: Path | None = None) -> pd.DataFrame:
    root = data_dir or require_data_dir()
    path = root / ENDO_FILE
    if not path.exists():
        raise FileNotFoundError(f"Endometriosis dataset not found: {path}\n{expected_data_paths()}")
    return pd.read_excel(path)


def train_pcos_bundle(df: pd.DataFrame) -> dict:
    x = df.drop("PCOS (Y/N)", axis=1)
    y = df["PCOS (Y/N)"]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    model.fit(x_scaled, y)
    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": list(x.columns),
    }


def train_endo_bundle(df: pd.DataFrame) -> dict:
    x = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    model.fit(x_scaled, y)
    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": list(x.columns),
    }


def save_bundles(pcos: dict, endo: dict, stats: dict | None = None) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pcos, MODELS_DIR / "pcos_bundle.joblib")
    joblib.dump(endo, MODELS_DIR / "endo_bundle.joblib")
    if stats is not None:
        joblib.dump(stats, MODELS_DIR / "feature_stats.joblib")


def load_bundles() -> tuple[dict, dict]:
    pcos_path = MODELS_DIR / "pcos_bundle.joblib"
    endo_path = MODELS_DIR / "endo_bundle.joblib"
    if not pcos_path.exists() or not endo_path.exists():
        raise FileNotFoundError("Saved models not found. Run training first.")
    return joblib.load(pcos_path), joblib.load(endo_path)


def build_feature_stats(data_dir: Path | None = None) -> dict:
    from feature_stats import ColumnStats, compute_dataset_stats

    root = data_dir or _resolve_data_dir()
    pcos_df = load_pcos_dataframe(root)
    endo_df = load_endo_dataframe(root)
    pcos_stats = compute_dataset_stats(pcos_df, target_col="PCOS (Y/N)")
    endo_stats = compute_dataset_stats(endo_df, target_col="Diagnosis")
    return {
        "pcos": {k: v.to_dict() for k, v in pcos_stats.items()},
        "endo": {k: v.to_dict() for k, v in endo_stats.items()},
    }


def load_feature_stats() -> dict:
    path = MODELS_DIR / "feature_stats.joblib"
    if not path.exists():
        raise FileNotFoundError("Feature stats not found. Retrain models first.")
    return joblib.load(path)


def get_feature_stats(data_dir: Path | None = None) -> dict:
    try:
        return load_feature_stats()
    except FileNotFoundError:
        return build_feature_stats(data_dir)


def train_and_save(data_dir: Path | None = None) -> tuple[dict, dict]:
    root = data_dir or _resolve_data_dir()
    pcos_df = load_pcos_dataframe(root)
    endo_df = load_endo_dataframe(root)
    pcos_bundle = train_pcos_bundle(pcos_df)
    endo_bundle = train_endo_bundle(endo_df)
    stats = build_feature_stats(root)
    save_bundles(pcos_bundle, endo_bundle, stats)
    return pcos_bundle, endo_bundle


def get_bundles(retrain: bool = False, data_dir: Path | None = None) -> tuple[dict, dict]:
    if not retrain:
        try:
            return load_bundles()
        except FileNotFoundError:
            pass
    return train_and_save(data_dir)


def predict_pcos_probability(bundle: dict, patient_df: pd.DataFrame) -> float:
    scaled = bundle["scaler"].transform(patient_df[bundle["feature_columns"]])
    return float(bundle["model"].predict_proba(scaled)[0][1])


def predict_endo_probability(bundle: dict, patient_df: pd.DataFrame) -> float:
    scaled = bundle["scaler"].transform(patient_df[bundle["feature_columns"]])
    return float(bundle["model"].predict_proba(scaled)[0][1])
