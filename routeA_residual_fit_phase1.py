from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_SET_VERSION = "legacy_aligned_backbone_v1"
MODEL_NAME = "logreg"
PRIMARY_PROTOCOL = "strict_amine_object_holdout"
CONFIRM_PROTOCOL = "ds_unknown"
TARGET_SYSTEM_TYPE = "single_amine_solvent"
TARGET_FAMILIES = ["alcohol", "sulfur_polar_aprotic"]
DIAGNOSTIC_ONLY_FAMILIES = ["glycol_ether_glyme", "other_organic_family"]
PRIMARY_SUCCESS_DELTA = 0.01
MIN_TRAIN_ROWS_FOR_LINEAR = 8
RIDGE_ALPHA = 1.0
PROB_CLIP_EPS = 1e-6
RESIDUAL_CLIP_MIN = -1.0
RESIDUAL_CLIP_MAX = 1.0

if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    ROOT = Path.cwd()
    if not (ROOT / "outputs").exists() and (ROOT.parent / "outputs").exists():
        ROOT = ROOT.parent
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"

BASE_FEATURES = [
    "ra_routed_main_delta_mean_avg",
    "ra_routed_main_delta_maxabs_avg",
    "ra_routed_main_delta_gap_avg",
    "ra_routed_aux_delta_mean_avg",
    "ra_routed_aux_delta_maxabs_avg",
    "ra_routed_aux_delta_gap_avg",
    "ra_has_carbamate_pair_any",
    "ra_has_bicarbonate_pair_any",
    "ra_reacted_form_count_avg",
    "ra_main_channel_carbamate_object_count",
    "ra_main_channel_protonation_object_count",
]


def load_routeA_table() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS_DIR / "routeA_family_aware_increment_ready_table.csv").copy()
    return df


def family_feature_cols(family: str) -> list[str]:
    return [f"fa_{family}__{feature}" for feature in BASE_FEATURES]


def compute_metrics(y_true: pd.Series, y_score: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    y_true = pd.Series(y_true).astype(int)
    y_score = pd.Series(y_score).astype(float)
    residual = y_true - y_score
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "support_rows": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "degenerate_single_class": int(y_true.nunique() < 2) if len(y_true) else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "f1": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "error_rate": float((y_pred != y_true).mean()) if len(y_true) else np.nan,
        "mean_residual": float(residual.mean()) if len(y_true) else np.nan,
        "mean_abs_residual": float(residual.abs().mean()) if len(y_true) else np.nan,
    }
    if len(y_true) == 0:
        return out
    if y_true.nunique() >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["average_precision"] = float(average_precision_score(y_true, y_score))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    out["f1"] = float(f1)
    out["precision"] = float(precision)
    out["recall"] = float(recall)
    return out


def delta_metric(corrected: float, baseline: float) -> float:
    if pd.isna(corrected) or pd.isna(baseline):
        return np.nan
    return float(corrected - baseline)


def fit_linear_residual_model(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline | None, str]:
    if len(x_train) < MIN_TRAIN_ROWS_FOR_LINEAR:
        return None, "intercept_only_low_support"
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)),
        ]
    )
    model.fit(x_train, y_train)
    return model, "ridge_linear_residual"


def select_scope(df: pd.DataFrame, protocol: str, family: str) -> pd.DataFrame:
    return df.loc[
        (df["protocol"] == protocol)
        & (df["routeA_family"] == family)
        & (df["system_type"] == TARGET_SYSTEM_TYPE)
        & (df["ra_row_increment_eligible_flag"] == 1)
    ].copy()


def run_family_protocol(df: pd.DataFrame, family: str, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = select_scope(df, protocol, family)
    features = family_feature_cols(family)
    prediction_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []

    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    fold_ids = sorted(int(fold) for fold in subset["fold_id"].dropna().unique())
    for fold_id in fold_ids:
        train_df = subset.loc[subset["fold_id"] != fold_id].copy()
        test_df = subset.loc[subset["fold_id"] == fold_id].copy()
        x_train = train_df[features].copy()
        y_train = train_df["residual"].astype(float).copy()
        x_test = test_df[features].copy()

        model, fit_mode = fit_linear_residual_model(x_train, y_train)
        if model is None:
            residual_hat = np.repeat(float(y_train.mean()) if len(y_train) else 0.0, len(test_df))
            coef_map = {feature: 0.0 for feature in features}
            intercept_value = float(y_train.mean()) if len(y_train) else 0.0
        else:
            residual_hat = model.predict(x_test)
            ridge = model.named_steps["ridge"]
            coef_map = dict(zip(features, ridge.coef_.tolist()))
            intercept_value = float(ridge.intercept_)

        residual_hat_raw = np.asarray(residual_hat, dtype=float)
        residual_hat = np.clip(residual_hat_raw, RESIDUAL_CLIP_MIN, RESIDUAL_CLIP_MAX)
        corrected_prob = np.clip(test_df["p_logreg"].astype(float).to_numpy() + residual_hat, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)

        for row_idx, row in enumerate(test_df.itertuples(index=False)):
            prediction_rows.append(
                {
                    "row_id": row.row_id,
                    "feature_set_version": FEATURE_SET_VERSION,
                    "baseline_model_name": MODEL_NAME,
                    "residual_model_name": "ridge_linear_residual",
                    "protocol": protocol,
                    "fold_id": int(fold_id),
                    "routeA_family": family,
                    "system_type": row.system_type,
                    "solvent_name_canonical": row.solvent_name_canonical,
                    "amine1_name_canonical": row.amine1_name_canonical,
                    "amine2_name_canonical": row.amine2_name_canonical,
                    "y_true": int(row.y_true),
                    "p_logreg_baseline": float(row.p_logreg),
                    "baseline_residual": float(row.residual),
                    "residual_target": float(row.residual),
                    "residual_hat_raw": float(residual_hat_raw[row_idx]),
                    "residual_hat": float(residual_hat[row_idx]),
                    "residual_hat_clipped_flag": int(abs(residual_hat_raw[row_idx] - residual_hat[row_idx]) > 1e-12),
                    "p_routeA_corrected": float(corrected_prob[row_idx]),
                    "corrected_residual": float(int(row.y_true) - corrected_prob[row_idx]),
                    "abs_baseline_residual": float(abs(row.residual)),
                    "abs_corrected_residual": float(abs(int(row.y_true) - corrected_prob[row_idx])),
                    "baseline_error_flag": int((float(row.p_logreg) >= 0.5) != int(row.y_true)),
                    "corrected_error_flag": int((corrected_prob[row_idx] >= 0.5) != int(row.y_true)),
                    "fit_mode": fit_mode,
                    "train_rows_used": int(len(train_df)),
                    "test_rows_used": int(len(test_df)),
                    "official_primary_evidence_row_flag": int(row.official_primary_evidence_row_flag),
                    "ds_unknown_confirmation_row_flag": int(row.ds_unknown_confirmation_row_flag),
                    "ra_reacted_form_count_avg": float(row.ra_reacted_form_count_avg),
                    "ra_main_channel_carbamate_object_count": int(row.ra_main_channel_carbamate_object_count),
                    "ra_main_channel_protonation_object_count": int(row.ra_main_channel_protonation_object_count),
                }
            )

        coef_record: dict[str, object] = {
            "feature_set_version": FEATURE_SET_VERSION,
            "baseline_model_name": MODEL_NAME,
            "residual_model_name": "ridge_linear_residual",
            "protocol": protocol,
            "routeA_family": family,
            "fold_id": int(fold_id),
            "fit_mode": fit_mode,
            "train_rows_used": int(len(train_df)),
            "test_rows_used": int(len(test_df)),
            "intercept": intercept_value,
        }
        coef_record.update({feature: float(value) for feature, value in coef_map.items()})
        coef_rows.append(coef_record)

    return pd.DataFrame(prediction_rows), pd.DataFrame(coef_rows)


def metric_rows_for_subset(pred_df: pd.DataFrame, family: str, protocol: str, fold_id: int | None, variant: str) -> dict[str, object]:
    subset = pred_df.loc[(pred_df["routeA_family"] == family) & (pred_df["protocol"] == protocol)].copy()
    if fold_id is not None:
        subset = subset.loc[subset["fold_id"] == fold_id].copy()
    y_true = subset["y_true"].astype(int)
    score_col = "p_logreg_baseline" if variant == "baseline_only" else "p_routeA_corrected"
    metrics = compute_metrics(y_true, subset[score_col])
    metrics.update(
        {
            "feature_set_version": FEATURE_SET_VERSION,
            "baseline_model_name": MODEL_NAME,
            "residual_model_name": "ridge_linear_residual",
            "protocol": protocol,
            "routeA_family": family,
            "fold_id": "all" if fold_id is None else int(fold_id),
            "variant": variant,
        }
    )
    return metrics


def build_fold_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in TARGET_FAMILIES:
        for protocol in [PRIMARY_PROTOCOL, CONFIRM_PROTOCOL]:
            subset = pred_df.loc[(pred_df["routeA_family"] == family) & (pred_df["protocol"] == protocol)].copy()
            if subset.empty:
                continue
            fold_ids = sorted(int(fold) for fold in subset["fold_id"].dropna().unique())
            for fold_id in fold_ids:
                for variant in ["baseline_only", "baseline_plus_residual"]:
                    rows.append(metric_rows_for_subset(subset, family, protocol, fold_id, variant))
    return pd.DataFrame(rows)


def summarize_coefficients(coef_df: pd.DataFrame, family: str, protocol: str) -> dict[str, object]:
    subset = coef_df.loc[(coef_df["routeA_family"] == family) & (coef_df["protocol"] == protocol)].copy()
    if subset.empty:
        return {
            "top_feature_1": "",
            "top_feature_1_mean_coef": np.nan,
            "top_feature_1_mean_abs_coef": np.nan,
            "top_feature_2": "",
            "top_feature_2_mean_coef": np.nan,
            "top_feature_2_mean_abs_coef": np.nan,
            "top_feature_3": "",
            "top_feature_3_mean_coef": np.nan,
            "top_feature_3_mean_abs_coef": np.nan,
        }
    summary = []
    for feature in family_feature_cols(family):
        mean_coef = float(subset[feature].mean())
        mean_abs_coef = float(subset[feature].abs().mean())
        summary.append((feature, mean_coef, mean_abs_coef))
    summary.sort(key=lambda item: item[2], reverse=True)
    out: dict[str, object] = {}
    for idx in range(3):
        feature, mean_coef, mean_abs_coef = summary[idx] if idx < len(summary) else ("", np.nan, np.nan)
        out[f"top_feature_{idx + 1}"] = feature
        out[f"top_feature_{idx + 1}_mean_coef"] = mean_coef
        out[f"top_feature_{idx + 1}_mean_abs_coef"] = mean_abs_coef
    return out


def build_summary_metrics(pred_df: pd.DataFrame, coef_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fold_metrics = build_fold_metrics(pred_df)
    for family in TARGET_FAMILIES:
        for protocol in [PRIMARY_PROTOCOL, CONFIRM_PROTOCOL]:
            subset = pred_df.loc[(pred_df["routeA_family"] == family) & (pred_df["protocol"] == protocol)].copy()
            if subset.empty:
                continue
            baseline = metric_rows_for_subset(subset, family, protocol, None, "baseline_only")
            corrected = metric_rows_for_subset(subset, family, protocol, None, "baseline_plus_residual")
            fold_subset = fold_metrics.loc[(fold_metrics["routeA_family"] == family) & (fold_metrics["protocol"] == protocol)].copy()
            baseline_folds = fold_subset.loc[fold_subset["variant"] == "baseline_only"].copy()
            corrected_folds = fold_subset.loc[fold_subset["variant"] == "baseline_plus_residual"].copy()

            row = {
                "feature_set_version": FEATURE_SET_VERSION,
                "baseline_model_name": MODEL_NAME,
                "residual_model_name": "ridge_linear_residual",
                "protocol": protocol,
                "routeA_family": family,
                "support_rows": int(corrected["support_rows"]),
                "n_pos": int(corrected["n_pos"]),
                "n_neg": int(corrected["n_neg"]),
                "baseline_roc_auc": baseline["roc_auc"],
                "corrected_roc_auc": corrected["roc_auc"],
                "delta_roc_auc": delta_metric(corrected["roc_auc"], baseline["roc_auc"]),
                "baseline_average_precision": baseline["average_precision"],
                "corrected_average_precision": corrected["average_precision"],
                "delta_average_precision": delta_metric(corrected["average_precision"], baseline["average_precision"]),
                "baseline_f1": baseline["f1"],
                "corrected_f1": corrected["f1"],
                "delta_f1": delta_metric(corrected["f1"], baseline["f1"]),
                "baseline_error_rate": baseline["error_rate"],
                "corrected_error_rate": corrected["error_rate"],
                "delta_error_rate": delta_metric(corrected["error_rate"], baseline["error_rate"]),
                "baseline_mean_residual": baseline["mean_residual"],
                "corrected_mean_residual": corrected["mean_residual"],
                "delta_mean_residual": delta_metric(corrected["mean_residual"], baseline["mean_residual"]),
                "baseline_mean_abs_residual": baseline["mean_abs_residual"],
                "corrected_mean_abs_residual": corrected["mean_abs_residual"],
                "delta_mean_abs_residual": delta_metric(corrected["mean_abs_residual"], baseline["mean_abs_residual"]),
                "fold_count": int(corrected_folds["fold_id"].nunique()),
                "baseline_fold_mean_roc_auc": float(baseline_folds["roc_auc"].mean()),
                "baseline_fold_std_roc_auc": float(baseline_folds["roc_auc"].std(ddof=1)) if len(baseline_folds) > 1 else np.nan,
                "corrected_fold_mean_roc_auc": float(corrected_folds["roc_auc"].mean()),
                "corrected_fold_std_roc_auc": float(corrected_folds["roc_auc"].std(ddof=1)) if len(corrected_folds) > 1 else np.nan,
                "baseline_fold_mean_average_precision": float(baseline_folds["average_precision"].mean()),
                "baseline_fold_std_average_precision": float(baseline_folds["average_precision"].std(ddof=1)) if len(baseline_folds) > 1 else np.nan,
                "corrected_fold_mean_average_precision": float(corrected_folds["average_precision"].mean()),
                "corrected_fold_std_average_precision": float(corrected_folds["average_precision"].std(ddof=1)) if len(corrected_folds) > 1 else np.nan,
                "baseline_fold_mean_error_rate": float(baseline_folds["error_rate"].mean()),
                "corrected_fold_mean_error_rate": float(corrected_folds["error_rate"].mean()),
                "fit_mode_set": "|".join(sorted(set(subset["fit_mode"]))),
            }
            row.update(summarize_coefficients(coef_df, family, protocol))
            rows.append(row)
    return pd.DataFrame(rows)


def build_confirmation_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    strict_lookup = (
        summary_df.loc[summary_df["protocol"] == PRIMARY_PROTOCOL, ["routeA_family", "delta_roc_auc", "delta_average_precision", "delta_error_rate"]]
        .rename(
            columns={
                "delta_roc_auc": "strict_delta_roc_auc",
                "delta_average_precision": "strict_delta_average_precision",
                "delta_error_rate": "strict_delta_error_rate",
            }
        )
        .copy()
    )
    confirm = summary_df.loc[summary_df["protocol"] == CONFIRM_PROTOCOL].copy()
    confirm = confirm.merge(strict_lookup, on="routeA_family", how="left", validate="one_to_one")
    confirm["direction_consistent_with_strict"] = (
        (
            (confirm["strict_delta_roc_auc"].fillna(0.0) >= 0.0) & (confirm["delta_roc_auc"].fillna(0.0) >= 0.0)
        )
        | (
            (confirm["strict_delta_average_precision"].fillna(0.0) >= 0.0) & (confirm["delta_average_precision"].fillna(0.0) >= 0.0)
        )
    ).astype(int)
    confirm["error_rate_not_worse_than_strict_direction"] = (
        confirm["delta_error_rate"].fillna(0.0) <= 1e-12
    ).astype(int)
    keep_cols = [
        "feature_set_version",
        "baseline_model_name",
        "residual_model_name",
        "routeA_family",
        "support_rows",
        "baseline_roc_auc",
        "corrected_roc_auc",
        "delta_roc_auc",
        "baseline_average_precision",
        "corrected_average_precision",
        "delta_average_precision",
        "baseline_error_rate",
        "corrected_error_rate",
        "delta_error_rate",
        "strict_delta_roc_auc",
        "strict_delta_average_precision",
        "strict_delta_error_rate",
        "direction_consistent_with_strict",
        "error_rate_not_worse_than_strict_direction",
    ]
    return confirm[keep_cols].copy()


def strict_success_flag(summary_row: pd.Series, confirm_row: pd.Series | None) -> str:
    primary_gain = (
        (pd.notna(summary_row["delta_roc_auc"]) and summary_row["delta_roc_auc"] >= PRIMARY_SUCCESS_DELTA)
        or (pd.notna(summary_row["delta_average_precision"]) and summary_row["delta_average_precision"] >= PRIMARY_SUCCESS_DELTA)
    )
    error_not_worse = pd.notna(summary_row["delta_error_rate"]) and summary_row["delta_error_rate"] <= 1e-12
    direction_ok = True
    if confirm_row is not None:
        direction_ok = bool(confirm_row["direction_consistent_with_strict"] == 1)
    if primary_gain and error_not_worse and direction_ok:
        return "established"
    if primary_gain and direction_ok:
        return "mixed"
    return "not_established"


def family_decision_table(summary_df: pd.DataFrame, confirm_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in TARGET_FAMILIES:
        strict_row = summary_df.loc[(summary_df["protocol"] == PRIMARY_PROTOCOL) & (summary_df["routeA_family"] == family)].iloc[0]
        confirm_match = confirm_df.loc[confirm_df["routeA_family"] == family]
        confirm_row = confirm_match.iloc[0] if not confirm_match.empty else None
        status = strict_success_flag(strict_row, confirm_row)
        rows.append(
            {
                "routeA_family": family,
                "strict_support_rows": int(strict_row["support_rows"]),
                "strict_delta_roc_auc": strict_row["delta_roc_auc"],
                "strict_delta_average_precision": strict_row["delta_average_precision"],
                "strict_delta_error_rate": strict_row["delta_error_rate"],
                "strict_delta_mean_abs_residual": strict_row["delta_mean_abs_residual"],
                "ds_unknown_delta_roc_auc": confirm_row["delta_roc_auc"] if confirm_row is not None else np.nan,
                "ds_unknown_delta_average_precision": confirm_row["delta_average_precision"] if confirm_row is not None else np.nan,
                "ds_unknown_delta_error_rate": confirm_row["delta_error_rate"] if confirm_row is not None else np.nan,
                "ds_unknown_direction_consistent": int(confirm_row["direction_consistent_with_strict"]) if confirm_row is not None else 0,
                "top_feature_1": strict_row["top_feature_1"],
                "top_feature_1_mean_coef": strict_row["top_feature_1_mean_coef"],
                "top_feature_2": strict_row["top_feature_2"],
                "top_feature_2_mean_coef": strict_row["top_feature_2_mean_coef"],
                "top_feature_3": strict_row["top_feature_3"],
                "top_feature_3_mean_coef": strict_row["top_feature_3_mean_coef"],
                "phase1_status": status,
            }
        )
    return pd.DataFrame(rows)


def build_all_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    coef_frames: list[pd.DataFrame] = []
    for family in TARGET_FAMILIES:
        for protocol in [PRIMARY_PROTOCOL, CONFIRM_PROTOCOL]:
            pred_df, coef_df = run_family_protocol(df, family, protocol)
            prediction_frames.append(pred_df)
            coef_frames.append(coef_df)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    coefficients = pd.concat(coef_frames, ignore_index=True)
    fold_metrics = build_fold_metrics(predictions)
    summary_metrics = build_summary_metrics(predictions, coefficients)
    confirmation = build_confirmation_table(summary_metrics)
    decision = family_decision_table(summary_metrics, confirmation)
    return predictions, fold_metrics, summary_metrics, confirmation, decision


def write_phase1_report(
    summary_df: pd.DataFrame,
    confirm_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> None:
    alcohol_strict = summary_df.loc[(summary_df["protocol"] == PRIMARY_PROTOCOL) & (summary_df["routeA_family"] == "alcohol")].iloc[0]
    sulfur_strict = summary_df.loc[(summary_df["protocol"] == PRIMARY_PROTOCOL) & (summary_df["routeA_family"] == "sulfur_polar_aprotic")].iloc[0]
    alcohol_confirm = confirm_df.loc[confirm_df["routeA_family"] == "alcohol"].iloc[0]
    sulfur_confirm = confirm_df.loc[confirm_df["routeA_family"] == "sulfur_polar_aprotic"].iloc[0]

    glycol_row = coverage_df.loc[
        (coverage_df["audit_scope"] == "protocol_family")
        & (coverage_df["protocol"] == PRIMARY_PROTOCOL)
        & (coverage_df["routeA_family"] == "glycol_ether_glyme")
    ].iloc[0]

    lines = [
        "# Route A Residual Fit Phase 1",
        "",
        "## Scope",
        "",
        "- Baseline stays frozen: `legacy_aligned_backbone_v1 + Logistic Regression`.",
        "- Residual target is `r = y_true - p_logreg`.",
        "- Only `strict_amine_object_holdout` is used for formal fitting; `DS-unknown` is used only as confirmation.",
        "- Scope is limited to `single_amine_solvent` rows in `alcohol` and `sulfur_polar_aprotic` with reacted-ready coverage.",
        "",
        "## Learner",
        "",
        f"- Residual learner: `Ridge(alpha={RIDGE_ALPHA})` with fold-internal imputation and standardization.",
        "- Reason: it is still a linear corrector, but numerically safer than raw OLS under small family-specific folds and correlated reacted summaries.",
        "- Predicted residuals are explicitly clipped to `[-1, 1]` before probability correction because the target residual itself cannot exceed that range.",
        "",
        "## Strict Results",
        "",
        (
            f"- alcohol: baseline ROC-AUC {alcohol_strict['baseline_roc_auc']:.3f} -> corrected {alcohol_strict['corrected_roc_auc']:.3f} "
            f"(delta {alcohol_strict['delta_roc_auc']:+.3f}); AP {alcohol_strict['baseline_average_precision']:.3f} -> "
            f"{alcohol_strict['corrected_average_precision']:.3f} (delta {alcohol_strict['delta_average_precision']:+.3f}); "
            f"error rate {alcohol_strict['baseline_error_rate']:.3f} -> {alcohol_strict['corrected_error_rate']:.3f} "
            f"(delta {alcohol_strict['delta_error_rate']:+.3f})."
        ),
        (
            f"- sulfur_polar_aprotic: baseline ROC-AUC {sulfur_strict['baseline_roc_auc']:.3f} -> corrected {sulfur_strict['corrected_roc_auc']:.3f} "
            f"(delta {sulfur_strict['delta_roc_auc']:+.3f}); AP {sulfur_strict['baseline_average_precision']:.3f} -> "
            f"{sulfur_strict['corrected_average_precision']:.3f} (delta {sulfur_strict['delta_average_precision']:+.3f}); "
            f"error rate {sulfur_strict['baseline_error_rate']:.3f} -> {sulfur_strict['corrected_error_rate']:.3f} "
            f"(delta {sulfur_strict['delta_error_rate']:+.3f})."
        ),
        "",
        "## DS-unknown Confirmation",
        "",
        (
            f"- alcohol confirmation: ROC-AUC delta {alcohol_confirm['delta_roc_auc']:+.3f}, AP delta {alcohol_confirm['delta_average_precision']:+.3f}, "
            f"error-rate delta {alcohol_confirm['delta_error_rate']:+.3f}, direction-consistent={int(alcohol_confirm['direction_consistent_with_strict'])}."
        ),
        (
            f"- sulfur_polar_aprotic confirmation: ROC-AUC delta {sulfur_confirm['delta_roc_auc']:+.3f}, AP delta {sulfur_confirm['delta_average_precision']:+.3f}, "
            f"error-rate delta {sulfur_confirm['delta_error_rate']:+.3f}, direction-consistent={int(sulfur_confirm['direction_consistent_with_strict'])}."
        ),
        "",
        "## Feature Signals",
        "",
    ]

    for row in decision_df.itertuples(index=False):
        lines.append(
            f"- {row.routeA_family}: top coefficients are `{row.top_feature_1}` ({row.top_feature_1_mean_coef:+.3f}), "
            f"`{row.top_feature_2}` ({row.top_feature_2_mean_coef:+.3f}), `{row.top_feature_3}` ({row.top_feature_3_mean_coef:+.3f}); "
            f"phase-1 status `{row.phase1_status}`."
        )

    lines.extend(
        [
            "",
            "## Why Glycol Ether / Glyme Stays Diagnostic",
            "",
            (
                f"- `glycol_ether_glyme` still has n={int(glycol_row['n_rows'])} strict rows and good reacted coverage "
                f"({glycol_row['row_increment_eligible_rate']:.1%}), but prior residual diagnosis showed a weaker or partly opposite bias direction than the main organic families."
            ),
            "- Merging it into the alcohol or sulfur residual branch would violate the family-aware Route A rule and risk averaging away branch-specific correction structure.",
        ]
    )

    (REPORTS_DIR / "routeA_residual_fit_phase1.md").write_text("\n".join(lines), encoding="utf-8")


def write_decision_report(decision_df: pd.DataFrame) -> None:
    alcohol = decision_df.loc[decision_df["routeA_family"] == "alcohol"].iloc[0]
    sulfur = decision_df.loc[decision_df["routeA_family"] == "sulfur_polar_aprotic"].iloc[0]

    if alcohol["phase1_status"] == "established" and sulfur["phase1_status"] == "established":
        overall = "expand_carefully"
        rationale = "Both priority families met the phase-1 success bar under strict and kept DS-unknown direction aligned."
    elif alcohol["phase1_status"] == "established" or sulfur["phase1_status"] == "established":
        overall = "selective_expand"
        rationale = "Only one priority family cleared the phase-1 bar, so Route A should expand only along the validated family branch."
    elif alcohol["phase1_status"] == "mixed" or sulfur["phase1_status"] == "mixed":
        overall = "hold_and_refine"
        rationale = "At least one family shows partial gain but not a clean, leakage-safe confirmation pattern yet."
    else:
        overall = "stop_phase1_expansion"
        rationale = "Neither priority family produced a clean enough phase-1 residual gain to justify broader Route A fitting."

    lines = [
        "# Route A Phase 1 Decision",
        "",
        "## Family Decisions",
        "",
        f"- alcohol: `{alcohol['phase1_status']}`",
        f"- sulfur_polar_aprotic: `{sulfur['phase1_status']}`",
        "",
        "## Next Step",
        "",
        f"- Overall Route A phase-1 decision: `{overall}`",
        f"- Reason: {rationale}",
        "- Keep `glycol_ether_glyme` as a diagnostic branch only in this phase.",
        "- Do not widen to `other_organic_family` or `dual_amine_solvent` until a family-specific branch is clearly established.",
    ]

    (REPORTS_DIR / "routeA_phase1_decision.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    routeA_df = load_routeA_table()
    coverage_df = pd.read_csv(OUTPUTS_DIR / "routeA_coverage_audit.csv")
    predictions, fold_metrics, summary_metrics, confirmation, decision = build_all_outputs(routeA_df)

    predictions.to_csv(OUTPUTS_DIR / "routeA_phase1_predictions.csv", index=False)
    summary_metrics.to_csv(OUTPUTS_DIR / "routeA_phase1_metrics.csv", index=False)
    fold_metrics.to_csv(OUTPUTS_DIR / "routeA_phase1_family_metrics.csv", index=False)
    confirmation.to_csv(OUTPUTS_DIR / "routeA_phase1_dsunknown_confirmation.csv", index=False)

    write_phase1_report(summary_metrics, confirmation, decision, coverage_df)
    write_decision_report(decision)


if __name__ == "__main__":
    main()
