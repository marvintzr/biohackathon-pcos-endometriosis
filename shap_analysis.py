"""SHAP analysis for the PCOS and endometriosis Random Forest models.

Computes per-feature contributions with TreeExplainer and writes, for each
model, a ranked feature table and a summary plot into shap_outputs/.

Usage:
    python shap_analysis.py

Requires the datasets in data/ (see README). Models are trained automatically
if models/ is empty.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files instead of opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ml_pipeline import (
    get_bundles,
    load_endo_dataframe,
    load_pcos_dataframe,
    require_data_dir,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "shap_outputs"
TOP_N = 15


def positive_class_shap(raw) -> np.ndarray:
    """Return an (n_samples, n_features) array of SHAP values for the positive class.

    SHAP returns different shapes for a binary classifier depending on version:
    either a list of two arrays (one per class), or a single array with a
    trailing class axis. Normalise both to the class-1 values.
    """
    if isinstance(raw, list):
        return np.asarray(raw[1] if len(raw) > 1 else raw[0])

    arr = np.asarray(raw)
    if arr.ndim == 3:
        return arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def analyse(name: str, bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Run TreeExplainer over one model and write its outputs. Returns the ranking."""
    features = bundle["feature_columns"]
    x = df[features]

    # The model was trained on scaled inputs, so SHAP must see scaled inputs too.
    x_scaled = bundle["scaler"].transform(x)

    print(f"\n[{name}] explaining {x_scaled.shape[0]} rows across {len(features)} features…")
    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = positive_class_shap(explainer.shap_values(x_scaled))

    # Mean absolute SHAP per feature = average magnitude of that feature's
    # influence on the prediction, regardless of direction.
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = (
        pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    table_path = OUTPUT_DIR / f"{name}_feature_importance.txt"
    with table_path.open("w", encoding="utf-8") as fh:
        fh.write(f"SHAP feature importance — {name} model\n")
        fh.write(f"Mean absolute SHAP value across {x_scaled.shape[0]} training rows\n\n")
        for i, row in ranking.iterrows():
            fh.write(f"{i + 1:>3}. {row['feature']:<40} {row['mean_abs_shap']:.4f}\n")
    print(f"[{name}] wrote {table_path.name}")

    # Beeswarm summary: each dot is one patient, position shows that feature's
    # push toward or away from a positive prediction.
    plt.figure()
    shap.summary_plot(
        shap_values,
        pd.DataFrame(x_scaled, columns=features),
        max_display=TOP_N,
        show=False,
    )
    plt.title(f"SHAP summary — {name} model")
    plt.tight_layout()
    beeswarm_path = OUTPUT_DIR / f"{name}_shap_summary.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{name}] wrote {beeswarm_path.name}")

    # Bar chart of the same ranking — easier to read in a README.
    plt.figure(figsize=(8, 6))
    top = ranking.head(TOP_N).iloc[::-1]
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {TOP_N} features — {name} model")
    plt.tight_layout()
    bar_path = OUTPUT_DIR / f"{name}_shap_importance.png"
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"[{name}] wrote {bar_path.name}")

    print(f"\n[{name}] top 10 features by mean |SHAP|:")
    for i, row in ranking.head(10).iterrows():
        print(f"  {i + 1:>2}. {row['feature']:<40} {row['mean_abs_shap']:.4f}")

    return ranking


def main() -> None:
    data_dir = require_data_dir()
    pcos_bundle, endo_bundle = get_bundles(data_dir=data_dir)

    analyse("pcos", pcos_bundle, load_pcos_dataframe(data_dir))
    analyse("endo", endo_bundle, load_endo_dataframe(data_dir))

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
