"""Streamlit app — PCOS vs Endometriosis (simple UI + medians for other features)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from feature_stats import (
    PCOS_FIELD_GROUPS,
    SIMPLE_ENDO_FIELDS,
    SIMPLE_PCOS_FIELDS,
    ColumnStats,
    merge_patient_row,
    out_of_range_warnings,
    unusual_value_warnings,
)
from ml_pipeline import (
    _resolve_data_dir,
    build_feature_stats,
    expected_data_paths,
    get_bundles,
    load_feature_stats,
    predict_endo_probability,
    predict_pcos_probability,
    require_data_dir,
    train_and_save,
)

THRESHOLD = 0.60

SIMPLE_LABELS: dict[str, str] = {
    "Age (yrs)": "Age (years)",
    "Weight (Kg)": "Weight (kg)",
    "Height(Cm)": "Height (cm)",
    "Cycle length(days)": "Cycle length (dataset scale 0–12)",
    "hair growth(Y/N)": "Excess hair growth (hirsutism)",
    "Weight gain(Y/N)": "Rapid weight gain",
    "Skin darkening (Y/N)": "Skin darkening",
    "Fast food (Y/N)": "Frequent fast food",
    "Follicle No. (R)": "Follicle count — right ovary",
    "Follicle No. (L)": "Follicle count — left ovary",
    "AMH(ng/mL)": "AMH (ng/mL)",
    "PRL(ng/mL)": "Prolactin (ng/mL)",
    "TSH (mIU/L)": "TSH (mIU/L)",
    "Chronic_Pain_Level": "Chronic pelvic pain (0–10)",
}


def _load_stats_raw(data_dir):
    try:
        return load_feature_stats()
    except FileNotFoundError:
        return build_feature_stats(data_dir)


def _stats_from_saved(raw: dict) -> dict[str, ColumnStats]:
    return {k: ColumnStats.from_dict(v) for k, v in raw.items()}


@st.cache_resource(show_spinner="Loading models and dataset statistics…")
def load_app_state():
    data_dir = require_data_dir()
    pcos_bundle, endo_bundle = get_bundles(data_dir=data_dir)
    stats_raw = _load_stats_raw(data_dir)
    return {
        "pcos_bundle": pcos_bundle,
        "endo_bundle": endo_bundle,
        "pcos_stats": _stats_from_saved(stats_raw["pcos"]),
        "endo_stats": _stats_from_saved(stats_raw["endo"]),
        "data_dir": data_dir,
    }


def _step(col: ColumnStats) -> float:
    if col.kind == "binary":
        return 1.0
    span = col.max_val - col.min_val
    if span <= 10:
        return 0.1 if span < 5 else 1.0
    if span <= 100:
        return 0.5
    return 1.0


def _render_field(
    col: ColumnStats,
    key_prefix: str,
    *,
    label: str | None = None,
) -> float:
    key = f"{key_prefix}_{col.name}"
    display = label or col.name
    help_text = (
        f"Training: min {col.min_val:g}, max {col.max_val:g}, "
        f"median {col.median:g} (typical {col.p05:g}–{col.p95:g})"
    )

    if col.kind in ("binary", "discrete"):
        labels = [str(int(v)) if float(v).is_integer() else str(v) for v in col.choices]
        default_idx = min(
            range(len(col.choices)),
            key=lambda i: abs(col.choices[i] - col.median),
        )
        choice = st.selectbox(display, labels, index=default_idx, key=key, help=help_text)
        return float(col.choices[labels.index(choice)])

    return float(
        st.number_input(
            display,
            min_value=float(col.min_val),
            max_value=float(col.max_val),
            value=float(col.median),
            step=_step(col),
            key=key,
            help=help_text,
        )
    )


def _render_field_list(
    fields: list[str],
    stats: dict[str, ColumnStats],
    key_prefix: str,
    *,
    use_friendly_labels: bool = False,
) -> dict[str, float]:
    values: dict[str, float] = {}
    cols = st.columns(2)
    for i, field in enumerate(fields):
        if field == "BMI" or field not in stats:
            continue
        with cols[i % 2]:
            label = SIMPLE_LABELS.get(field) if use_friendly_labels else None
            values[field] = _render_field(stats[field], key_prefix, label=label)
    return values


def _render_pcos_fields(
    pcos_stats: dict[str, ColumnStats],
    columns: list[str],
    fields: list[str],
    key_prefix: str,
    *,
    use_friendly_labels: bool = False,
) -> dict[str, float]:
    ordered = [f for f in fields if f in columns and f in pcos_stats]
    values = _render_field_list(ordered, pcos_stats, key_prefix, use_friendly_labels=use_friendly_labels)

    if "Weight (Kg)" in values and "Height(Cm)" in values:
        h_m = values["Height(Cm)"] / 100.0
        values["BMI"] = (
            values["Weight (Kg)"] / (h_m * h_m) if h_m > 0 else pcos_stats["BMI"].median
        )
        st.caption(f"**BMI** (from weight & height): {values['BMI']:.2f}")
    return values


def _render_advanced_pcos(
    pcos_stats: dict[str, ColumnStats],
    columns: list[str],
    simple_fields: set[str],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for group_name, group_fields in PCOS_FIELD_GROUPS.items():
        extra = [f for f in group_fields if f in columns and f in pcos_stats and f not in simple_fields]
        if not extra:
            continue
        with st.expander(group_name, expanded=False):
            values.update(_render_field_list(extra, pcos_stats, f"adv_pcos"))
    return values


def _render_advanced_endo(
    endo_stats: dict[str, ColumnStats],
    columns: list[str],
    simple_fields: set[str],
) -> dict[str, float]:
    extra = [f for f in columns if f in endo_stats and f not in simple_fields]
    if not extra:
        return {}
    values: dict[str, float] = {}
    with st.expander("All endometriosis model fields", expanded=False):
        values.update(_render_field_list(extra, endo_stats, "adv_endo"))
    return values


def _sync_endo_from_pcos(
    pcos_values: dict[str, float],
    endo_values: dict[str, float],
    endo_overrides: dict[str, float],
) -> dict[str, float]:
    out = dict(endo_values)
    if "Age (yrs)" in pcos_values and "Age" in out:
        out["Age"] = pcos_values["Age (yrs)"]
    if "BMI" in pcos_values and "BMI" in out:
        out["BMI"] = pcos_values["BMI"]
    if "Cycle length(days)" in pcos_values and "Menstrual_Irregularity" in out:
        if "Menstrual_Irregularity" not in endo_overrides:
            out["Menstrual_Irregularity"] = float(pcos_values["Cycle length(days)"] > 5)
    return out


def main():
    st.set_page_config(
        page_title="PCOS & Endo Diagnostician",
        page_icon="🩺",
        layout="wide",
    )
    st.title("Differential Diagnostician: PCOS vs Endometriosis")
    st.caption(
        "Simple mode: key clinical inputs only. All other model features use training medians. "
        "For education only."
    )

    try:
        state = load_app_state()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.markdown(
            "Put both Excel files in the **`data`** folder inside this project:\n\n"
            f"```\n{expected_data_paths()}```"
        )
        return

    pcos_bundle = state["pcos_bundle"]
    endo_bundle = state["endo_bundle"]
    pcos_stats = state["pcos_stats"]
    endo_stats = state["endo_stats"]
    data_dir = state["data_dir"]
    pcos_columns = pcos_bundle["feature_columns"]
    endo_columns = endo_bundle["feature_columns"]
    simple_pcos_set = set(SIMPLE_PCOS_FIELDS)
    simple_endo_set = set(SIMPLE_ENDO_FIELDS)
    n_user_pcos = len([f for f in SIMPLE_PCOS_FIELDS if f in pcos_columns])
    if "BMI" in pcos_columns and "Weight (Kg)" in SIMPLE_PCOS_FIELDS:
        n_user_pcos += 1  # BMI derived from weight & height
    n_median_pcos = len(pcos_columns) - n_user_pcos

    st.sidebar.header("Session")
    st.sidebar.caption(f"Data: `{data_dir}`")
    st.sidebar.caption(f"Model uses {len(pcos_columns)} PCOS features")
    st.sidebar.caption(f"Simple form: {len(SIMPLE_PCOS_FIELDS)} + BMI (auto)")
    if st.sidebar.button("Retrain models & refresh stats"):
        st.cache_resource.clear()
        train_and_save(data_dir)
        st.sidebar.success("Done.")
        st.rerun()

    with st.form("patient_form"):
        st.subheader("Clinical inputs")
        st.info(
            f"**{n_median_pcos}** other PCOS features are set to **training medians** automatically "
            "(not shown). Expand **Advanced** only if you need full control."
        )

        pcos_overrides = _render_pcos_fields(
            pcos_stats,
            pcos_columns,
            SIMPLE_PCOS_FIELDS,
            "simple_pcos",
            use_friendly_labels=True,
        )

        st.markdown("**Layer 2 — endometriosis**")
        endo_overrides = _render_field_list(
            [f for f in SIMPLE_ENDO_FIELDS if f in endo_columns],
            endo_stats,
            "simple_endo",
            use_friendly_labels=True,
        )
        st.caption("Age and BMI for the endometriosis model are taken from the fields above.")

        with st.expander("Advanced — all 41 PCOS features + full endo inputs", expanded=False):
            pcos_overrides.update(
                _render_advanced_pcos(pcos_stats, pcos_columns, simple_pcos_set | {"BMI"})
            )
            endo_overrides.update(
                _render_advanced_endo(endo_stats, endo_columns, simple_endo_set)
            )

        submitted = st.form_submit_button("Run diagnostic pathway", type="primary")

    if not submitted:
        return

    pcos_row = merge_patient_row(pcos_columns, pcos_stats, pcos_overrides)
    pcos_values = pcos_row.iloc[0].to_dict()

    endo_row = merge_patient_row(endo_columns, endo_stats, endo_overrides)
    endo_values = _sync_endo_from_pcos(pcos_values, endo_row.iloc[0].to_dict(), endo_overrides)
    endo_row = pd.DataFrame([endo_values], columns=endo_columns)

    all_overrides = {**pcos_overrides, **endo_overrides}
    range_warnings = out_of_range_warnings(all_overrides, {**pcos_stats, **endo_stats})
    unusual = unusual_value_warnings(all_overrides, {**pcos_stats, **endo_stats})

    if range_warnings:
        st.error("**Outside training range** (for fields you entered):")
        for w in range_warnings:
            st.markdown(f"- {w}")
    if unusual:
        st.warning("**Uncommon values** (for fields you entered):")
        for w in unusual:
            st.markdown(f"- {w}")
    if not range_warnings and not unusual:
        st.success("Entered values look consistent with the training data.")

    with st.expander("See full feature vector sent to the model (including medians)"):
        st.dataframe(pcos_row.T.rename(columns={0: "value"}), use_container_width=True)

    pcos_prob = predict_pcos_probability(pcos_bundle, pcos_row)
    st.markdown("### Clinical summary")
    st.progress(
        min(max(pcos_prob, 0.0), 1.0),
        text=f"Layer 1 — PCOS probability: {pcos_prob * 100:.1f}%",
    )

    if pcos_prob > THRESHOLD:
        st.success(
            f"**Layer 1: PCOS likely** ({pcos_prob * 100:.1f}% confidence). "
            "Patterns align with Rotterdam criteria."
        )
        return

    st.write(f"Layer 1: PCOS unlikely ({pcos_prob * 100:.1f}%). Checking layer 2…")
    endo_prob = predict_endo_probability(endo_bundle, endo_row)
    st.progress(
        min(max(endo_prob, 0.0), 1.0),
        text=f"Layer 2 — Endometriosis probability: {endo_prob * 100:.1f}%",
    )

    if endo_prob > THRESHOLD:
        st.info(
            f"**Layer 2: Endometriosis possible** ({endo_prob * 100:.1f}% confidence)."
        )
        return

    st.write(f"Layer 2: Endometriosis unlikely ({endo_prob * 100:.1f}%). Layer 3 rule-outs…")
    tsh = pcos_values.get("TSH (mIU/L)", pcos_stats["TSH (mIU/L)"].median)
    prl = pcos_values.get("PRL(ng/mL)", pcos_stats["PRL(ng/mL)"].median)

    flags = []
    if tsh > 4.5:
        flags.append(f"Elevated TSH ({tsh:.2f}) — consider hypothyroidism workup.")
    elif tsh < 0.4:
        flags.append(f"Low TSH ({tsh:.2f}) — consider hyperthyroidism workup.")
    if prl > 25:
        flags.append(f"Elevated prolactin ({prl:.1f}) — consider hyperprolactinemia workup.")

    if flags:
        for msg in flags:
            st.error(msg)
    else:
        st.success("TSH and prolactin within typical screening range.")
        st.warning("Refer to reproductive endocrinology for comprehensive evaluation.")

    st.divider()
    st.caption("Decision support only — not a medical diagnosis.")


if __name__ == "__main__":
    main()
