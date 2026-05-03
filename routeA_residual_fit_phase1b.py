from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_SET_VERSION = "legacy_aligned_backbone_v1"
BASELINE_MODEL_NAME = "logreg"
RESIDUAL_MODEL_NAME = "ridge_linear_residual_phase1b"
PRIMARY_PROTOCOL = "strict_amine_object_holdout"
CONFIRM_PROTOCOL = "ds_unknown"
TARGET_SYSTEM_TYPE = "single_amine_solvent"
TARGET_FAMILIES = ["alcohol", "sulfur_polar_aprotic"]
PRIMARY_SUCCESS_DELTA = 0.01
MIN_TRAIN_ROWS_FOR_LINEAR = 8
PROB_CLIP_EPS = 1e-6
STRONGER_ALPHA_BY_FAMILY = {
    "alcohol": 10.0,
    "sulfur_polar_aprotic": 25.0,
}
FEATURES_BY_FAMILY = {
    "alcohol": [
        "fa_alcohol__ra_routed_main_delta_gap_avg",
        "fa_alcohol__ra_routed_main_delta_maxabs_avg",
        "fa_alcohol__ra_routed_main_delta_mean_avg",
    ],
    "sulfur_polar_aprotic": [
        "fa_sulfur_polar_aprotic__ra_routed_main_delta_gap_avg",
        "fa_sulfur_polar_aprotic__ra_routed_main_delta_mean_avg",
    ],
}

if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    ROOT = Path.cwd()
    if not (ROOT / "outputs").exists() and (ROOT.parent / "outputs").exists():
        ROOT = ROOT.parent
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"

def load_routeA_table() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS_DIR / "routeA_family_aware_increment_ready_table.csv").copy()
    return df


def family_feature_cols(family: str) -> list[str]:
    return FEATURES_BY_FAMILY[family]


def compute_metrics(y_true: pd.Series, y_score: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    y_true = pd.Series(y_true).astype(int)
    y_score = pd.Series(y_score).astype(float).clip(PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)
    residual = y_true - y_score
    y_pred = (y_score >= threshold).astype(int)
    positive_rate = float(y_true.mean()) if len(y_true) else np.nan
    pred_mean = float(y_score.mean()) if len(y_true) else np.nan
    out = {
        "support_rows": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "positive_rate": positive_rate,
        "degenerate_single_class": int(y_true.nunique() < 2) if len(y_true) else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "f1": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "error_rate": float((y_pred != y_true).mean()) if len(y_true) else np.nan,
        "mean_residual": float(residual.mean()) if len(y_true) else np.nan,
        "mean_abs_residual": float(residual.abs().mean()) if len(y_true) else np.nan,
        "brier_score": float(brier_score_loss(y_true, y_score)) if len(y_true) else np.nan,
        "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])) if len(y_true) else np.nan,
        "pred_mean": pred_mean,
        "calibration_drift": float(pred_mean - positive_rate) if len(y_true) else np.nan,
        "abs_calibration_drift": float(abs(pred_mean - positive_rate)) if len(y_true) else np.nan,
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


def fit_linear_residual_model(x_train: pd.DataFrame, y_train: pd.Series, family: str) -> tuple[Pipeline | None, str]:
    if len(x_train) < MIN_TRAIN_ROWS_FOR_LINEAR:
        return None, "intercept_only_low_support"
    alpha = STRONGER_ALPHA_BY_FAMILY[family]
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
        ]
    )
    model.fit(x_train, y_train)
    return model, f"ridge_linear_residual_alpha_{alpha:g}"


def feasible_corrected_prob(base_prob: np.ndarray, raw_residual_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = -base_prob
    upper = 1.0 - base_prob
    feasible_residual_hat = np.clip(raw_residual_hat, lower, upper)
    corrected_prob = np.clip(base_prob + feasible_residual_hat, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)
    clipped_flag = (np.abs(feasible_residual_hat - raw_residual_hat) > 1e-12).astype(int)
    return feasible_residual_hat, corrected_prob, clipped_flag


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

        model, fit_mode = fit_linear_residual_model(x_train, y_train, family)
        if model is None:
            residual_hat_raw = np.repeat(float(y_train.mean()) if len(y_train) else 0.0, len(test_df))
            coef_map = {feature: 0.0 for feature in features}
            intercept_value = float(y_train.mean()) if len(y_train) else 0.0
        else:
            residual_hat_raw = model.predict(x_test)
            ridge = model.named_steps["ridge"]
            coef_map = dict(zip(features, ridge.coef_.tolist()))
            intercept_value = float(ridge.intercept_)

        base_prob = test_df["p_logreg"].astype(float).to_numpy()
        residual_hat_raw = np.asarray(residual_hat_raw, dtype=float)
        residual_hat, corrected_prob, clipped_flag = feasible_corrected_prob(base_prob, residual_hat_raw)

        for row_idx, row in enumerate(test_df.itertuples(index=False)):
            prediction_rows.append(
                {
                    "row_id": row.row_id,
                    "feature_set_version": FEATURE_SET_VERSION,
                    "baseline_model_name": BASELINE_MODEL_NAME,
                    "residual_model_name": RESIDUAL_MODEL_NAME,
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
                    "residual_hat_feasible_clip_flag": int(clipped_flag[row_idx]),
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
                }
            )

        coef_record: dict[str, object] = {
            "feature_set_version": FEATURE_SET_VERSION,
            "baseline_model_name": BASELINE_MODEL_NAME,
            "residual_model_name": RESIDUAL_MODEL_NAME,
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
    score_col = "p_logreg_baseline" if variant == "baseline_only" else "p_routeA_corrected"
    metrics = compute_metrics(subset["y_true"], subset[score_col])
    metrics.update(
        {
            "feature_set_version": FEATURE_SET_VERSION,
            "baseline_model_name": BASELINE_MODEL_NAME,
            "residual_model_name": RESIDUAL_MODEL_NAME,
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
        return {}
    summary = []
    for feature in family_feature_cols(family):
        mean_coef = float(subset[feature].mean())
        mean_abs_coef = float(subset[feature].abs().mean())
        summary.append((feature, mean_coef, mean_abs_coef))
    summary.sort(key=lambda item: item[2], reverse=True)
    out: dict[str, object] = {}
    for idx, item in enumerate(summary, start=1):
        feature, mean_coef, mean_abs_coef = item
        out[f"top_feature_{idx}"] = feature
        out[f"top_feature_{idx}_mean_coef"] = mean_coef
        out[f"top_feature_{idx}_mean_abs_coef"] = mean_abs_coef
    return out


def classify_improvement(row: pd.Series) -> str:
    ranking_gain = (
        (pd.notna(row["delta_roc_auc"]) and row["delta_roc_auc"] >= PRIMARY_SUCCESS_DELTA)
        or (pd.notna(row["delta_average_precision"]) and row["delta_average_precision"] >= PRIMARY_SUCCESS_DELTA)
    )
    calibration_gain = (
        (pd.notna(row["delta_brier_score"]) and row["delta_brier_score"] < 0.0)
        or (pd.notna(row["delta_log_loss"]) and row["delta_log_loss"] < 0.0)
        or (pd.notna(row["delta_abs_calibration_drift"]) and row["delta_abs_calibration_drift"] < 0.0)
        or (pd.notna(row["delta_error_rate"]) and row["delta_error_rate"] < 0.0)
    )
    if ranking_gain and calibration_gain:
        return "ranking_plus_calibration"
    if ranking_gain:
        return "ranking_only"
    if calibration_gain:
        return "calibration_or_threshold_only"
    return "no_gain_or_degradation"


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
                "baseline_model_name": BASELINE_MODEL_NAME,
                "residual_model_name": RESIDUAL_MODEL_NAME,
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
                "baseline_brier_score": baseline["brier_score"],
                "corrected_brier_score": corrected["brier_score"],
                "delta_brier_score": delta_metric(corrected["brier_score"], baseline["brier_score"]),
                "baseline_log_loss": baseline["log_loss"],
                "corrected_log_loss": corrected["log_loss"],
                "delta_log_loss": delta_metric(corrected["log_loss"], baseline["log_loss"]),
                "baseline_pred_mean": baseline["pred_mean"],
                "corrected_pred_mean": corrected["pred_mean"],
                "delta_pred_mean": delta_metric(corrected["pred_mean"], baseline["pred_mean"]),
                "baseline_calibration_drift": baseline["calibration_drift"],
                "corrected_calibration_drift": corrected["calibration_drift"],
                "delta_calibration_drift": delta_metric(corrected["calibration_drift"], baseline["calibration_drift"]),
                "baseline_abs_calibration_drift": baseline["abs_calibration_drift"],
                "corrected_abs_calibration_drift": corrected["abs_calibration_drift"],
                "delta_abs_calibration_drift": delta_metric(corrected["abs_calibration_drift"], baseline["abs_calibration_drift"]),
                "baseline_mean_residual": baseline["mean_residual"],
                "corrected_mean_residual": corrected["mean_residual"],
                "delta_mean_residual": delta_metric(corrected["mean_residual"], baseline["mean_residual"]),
                "baseline_mean_abs_residual": baseline["mean_abs_residual"],
                "corrected_mean_abs_residual": corrected["mean_abs_residual"],
                "delta_mean_abs_residual": delta_metric(corrected["mean_abs_residual"], baseline["mean_abs_residual"]),
                "pred_mean_shift": delta_metric(corrected["pred_mean"], baseline["pred_mean"]),
                "feasible_clip_rate": float(subset["residual_hat_feasible_clip_flag"].mean()),
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
            row["improvement_mode"] = classify_improvement(pd.Series(row))
            rows.append(row)
    return pd.DataFrame(rows)


def build_confirmation_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    strict_lookup = (
        summary_df.loc[
            summary_df["protocol"] == PRIMARY_PROTOCOL,
            ["routeA_family", "delta_roc_auc", "delta_average_precision", "delta_error_rate", "delta_brier_score", "delta_log_loss", "improvement_mode"],
        ]
        .rename(
            columns={
                "delta_roc_auc": "strict_delta_roc_auc",
                "delta_average_precision": "strict_delta_average_precision",
                "delta_error_rate": "strict_delta_error_rate",
                "delta_brier_score": "strict_delta_brier_score",
                "delta_log_loss": "strict_delta_log_loss",
                "improvement_mode": "strict_improvement_mode",
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
    confirm["calibration_not_worse_than_strict"] = (
        (confirm["delta_brier_score"].fillna(0.0) <= 1e-12)
        & (confirm["delta_log_loss"].fillna(0.0) <= 1e-12)
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
        "baseline_brier_score",
        "corrected_brier_score",
        "delta_brier_score",
        "baseline_log_loss",
        "corrected_log_loss",
        "delta_log_loss",
        "strict_delta_roc_auc",
        "strict_delta_average_precision",
        "strict_delta_error_rate",
        "strict_delta_brier_score",
        "strict_delta_log_loss",
        "direction_consistent_with_strict",
        "calibration_not_worse_than_strict",
        "improvement_mode",
        "strict_improvement_mode",
    ]
    return confirm[keep_cols].copy()


def phase1b_status(strict_row: pd.Series, confirm_row: pd.Series | None) -> str:
    ranking_gain = (
        (pd.notna(strict_row["delta_roc_auc"]) and strict_row["delta_roc_auc"] >= PRIMARY_SUCCESS_DELTA)
        or (pd.notna(strict_row["delta_average_precision"]) and strict_row["delta_average_precision"] >= PRIMARY_SUCCESS_DELTA)
    )
    error_not_worse = pd.notna(strict_row["delta_error_rate"]) and strict_row["delta_error_rate"] <= 1e-12
    calibration_not_worse = (
        pd.notna(strict_row["delta_brier_score"])
        and pd.notna(strict_row["delta_log_loss"])
        and strict_row["delta_brier_score"] <= 1e-12
        and strict_row["delta_log_loss"] <= 1e-12
    )
    not_threshold_only = strict_row["improvement_mode"] in {"ranking_plus_calibration", "ranking_only"}
    direction_ok = True
    if confirm_row is not None:
        direction_ok = bool(confirm_row["direction_consistent_with_strict"] == 1)
    if ranking_gain and error_not_worse and calibration_not_worse and direction_ok and not_threshold_only:
        return "established"
    if error_not_worse and calibration_not_worse and not ranking_gain:
        return "calibration_only"
    return "not_established"


def build_decision_table(summary_df: pd.DataFrame, confirm_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in TARGET_FAMILIES:
        strict_row = summary_df.loc[(summary_df["protocol"] == PRIMARY_PROTOCOL) & (summary_df["routeA_family"] == family)].iloc[0]
        confirm_match = confirm_df.loc[confirm_df["routeA_family"] == family]
        confirm_row = confirm_match.iloc[0] if not confirm_match.empty else None
        status = phase1b_status(strict_row, confirm_row)
        rows.append(
            {
                "routeA_family": family,
                "strict_support_rows": int(strict_row["support_rows"]),
                "strict_delta_roc_auc": strict_row["delta_roc_auc"],
                "strict_delta_average_precision": strict_row["delta_average_precision"],
                "strict_delta_error_rate": strict_row["delta_error_rate"],
                "strict_delta_brier_score": strict_row["delta_brier_score"],
                "strict_delta_log_loss": strict_row["delta_log_loss"],
                "strict_pred_mean_shift": strict_row["pred_mean_shift"],
                "strict_improvement_mode": strict_row["improvement_mode"],
                "ds_unknown_delta_roc_auc": confirm_row["delta_roc_auc"] if confirm_row is not None else np.nan,
                "ds_unknown_delta_average_precision": confirm_row["delta_average_precision"] if confirm_row is not None else np.nan,
                "ds_unknown_delta_error_rate": confirm_row["delta_error_rate"] if confirm_row is not None else np.nan,
                "ds_unknown_delta_brier_score": confirm_row["delta_brier_score"] if confirm_row is not None else np.nan,
                "ds_unknown_delta_log_loss": confirm_row["delta_log_loss"] if confirm_row is not None else np.nan,
                "ds_unknown_direction_consistent": int(confirm_row["direction_consistent_with_strict"]) if confirm_row is not None else 0,
                "top_feature_1": strict_row.get("top_feature_1", ""),
                "top_feature_1_mean_coef": strict_row.get("top_feature_1_mean_coef", np.nan),
                "top_feature_2": strict_row.get("top_feature_2", ""),
                "top_feature_2_mean_coef": strict_row.get("top_feature_2_mean_coef", np.nan),
                "top_feature_3": strict_row.get("top_feature_3", ""),
                "top_feature_3_mean_coef": strict_row.get("top_feature_3_mean_coef", np.nan),
                "phase1b_status": status,
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
    decision = build_decision_table(summary_metrics, confirmation)
    return predictions, fold_metrics, summary_metrics, confirmation, decision


def write_phase1b_report(summary_df: pd.DataFrame, confirm_df: pd.DataFrame, decision_df: pd.DataFrame) -> None:
    alcohol_strict = summary_df.loc[(summary_df["protocol"] == PRIMARY_PROTOCOL) & (summary_df["routeA_family"] == "alcohol")].iloc[0]
    sulfur_strict = summary_df.loc[(summary_df["protocol"] == PRIMARY_PROTOCOL) & (summary_df["routeA_family"] == "sulfur_polar_aprotic")].iloc[0]
    alcohol_confirm = confirm_df.loc[confirm_df["routeA_family"] == "alcohol"].iloc[0]
    sulfur_confirm = confirm_df.loc[confirm_df["routeA_family"] == "sulfur_polar_aprotic"].iloc[0]

    lines = [
        "# Route A Residual Fit Phase 1B",
        "",
        "## Scope",
        "",
        "- Baseline remains frozen.",
        "- Scope remains restricted to `strict + single_amine_solvent + alcohol` and `strict + single_amine_solvent + sulfur_polar_aprotic`.",
        "- `DS-unknown` is confirmation only.",
        "",
        "## Minimal Repair",
        "",
        "- Each family keeps only the strongest main-routed reacted-delta features.",
        "- `alcohol`: `main_delta_gap`, `main_delta_maxabs`, `main_delta_mean`.",
        "- `sulfur_polar_aprotic`: `main_delta_gap`, `main_delta_mean`.",
        "- Linear residual corrector is kept, but regularization is strengthened and correction is clipped to the row-wise feasible residual interval `[-p_base, 1 - p_base]`.",
        "",
        "## Strict Results",
        "",
        (
            f"- alcohol: baseline ROC-AUC {alcohol_strict['baseline_roc_auc']:.3f} -> corrected {alcohol_strict['corrected_roc_auc']:.3f} "
            f"(delta {alcohol_strict['delta_roc_auc']:+.3f}); AP {alcohol_strict['baseline_average_precision']:.3f} -> "
            f"{alcohol_strict['corrected_average_precision']:.3f} (delta {alcohol_strict['delta_average_precision']:+.3f}); "
            f"Brier {alcohol_strict['baseline_brier_score']:.3f} -> {alcohol_strict['corrected_brier_score']:.3f}; "
            f"log loss {alcohol_strict['baseline_log_loss']:.3f} -> {alcohol_strict['corrected_log_loss']:.3f}; "
            f"pred_mean shift {alcohol_strict['pred_mean_shift']:+.3f}; mode `{alcohol_strict['improvement_mode']}`."
        ),
        (
            f"- sulfur_polar_aprotic: baseline ROC-AUC {sulfur_strict['baseline_roc_auc']:.3f} -> corrected {sulfur_strict['corrected_roc_auc']:.3f} "
            f"(delta {sulfur_strict['delta_roc_auc']:+.3f}); AP {sulfur_strict['baseline_average_precision']:.3f} -> "
            f"{sulfur_strict['corrected_average_precision']:.3f} (delta {sulfur_strict['delta_average_precision']:+.3f}); "
            f"Brier {sulfur_strict['baseline_brier_score']:.3f} -> {sulfur_strict['corrected_brier_score']:.3f}; "
            f"log loss {sulfur_strict['baseline_log_loss']:.3f} -> {sulfur_strict['corrected_log_loss']:.3f}; "
            f"pred_mean shift {sulfur_strict['pred_mean_shift']:+.3f}; mode `{sulfur_strict['improvement_mode']}`."
        ),
        "",
        "## DS-unknown Confirmation",
        "",
        (
            f"- alcohol confirmation: ROC delta {alcohol_confirm['delta_roc_auc']:+.3f}, AP delta {alcohol_confirm['delta_average_precision']:+.3f}, "
            f"Brier delta {alcohol_confirm['delta_brier_score']:+.3f}, log-loss delta {alcohol_confirm['delta_log_loss']:+.3f}, "
            f"direction-consistent={int(alcohol_confirm['direction_consistent_with_strict'])}."
        ),
        (
            f"- sulfur_polar_aprotic confirmation: ROC delta {sulfur_confirm['delta_roc_auc']:+.3f}, AP delta {sulfur_confirm['delta_average_precision']:+.3f}, "
            f"Brier delta {sulfur_confirm['delta_brier_score']:+.3f}, log-loss delta {sulfur_confirm['delta_log_loss']:+.3f}, "
            f"direction-consistent={int(sulfur_confirm['direction_consistent_with_strict'])}."
        ),
        "",
        "## Feature Signals",
        "",
    ]

    for row in decision_df.itertuples(index=False):
        lines.append(
            f"- {row.routeA_family}: `{row.top_feature_1}` ({row.top_feature_1_mean_coef:+.3f}), "
            f"`{row.top_feature_2}` ({row.top_feature_2_mean_coef:+.3f}), `{row.top_feature_3}` ({row.top_feature_3_mean_coef:+.3f}) -> "
            f"status `{row.phase1b_status}`."
        )

    (REPORTS_DIR / "routeA_residual_fit_phase1b.md").write_text("\n".join(lines), encoding="utf-8")


def write_phase1b_decision(decision_df: pd.DataFrame) -> None:
    alcohol = decision_df.loc[decision_df["routeA_family"] == "alcohol"].iloc[0]
    sulfur = decision_df.loc[decision_df["routeA_family"] == "sulfur_polar_aprotic"].iloc[0]

    if alcohol["phase1b_status"] == "established" or sulfur["phase1b_status"] == "established":
        overall = "recover_to_go"
        rationale = "At least one priority family shows a ranking-level gain under strict without error-rate or calibration deterioration, and DS-unknown is not reversing the direction."
    else:
        overall = "stop_after_phase1b"
        rationale = "No family produced the required ranking improvement under strict with non-reversing DS-unknown confirmation and stable calibration. Any remaining gain is calibration-only and is insufficient for continued Route A expansion."

    lines = [
        "# Route A Phase 1B Decision",
        "",
        "## Family Decisions",
        "",
        f"- alcohol: `{alcohol['phase1b_status']}`",
        f"- sulfur_polar_aprotic: `{sulfur['phase1b_status']}`",
        "",
        "## Next Step",
        "",
        f"- Route A outcome after phase 1b: `{overall}`",
        f"- Reason: {rationale}",
        "- No family scope expansion is justified beyond this point in phase 1b.",
    ]

    (REPORTS_DIR / "routeA_phase1b_decision.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    routeA_df = load_routeA_table()
    predictions, fold_metrics, summary_metrics, confirmation, decision = build_all_outputs(routeA_df)

    predictions.to_csv(OUTPUTS_DIR / "routeA_phase1b_predictions.csv", index=False)
    summary_metrics.to_csv(OUTPUTS_DIR / "routeA_phase1b_metrics.csv", index=False)
    fold_metrics.to_csv(OUTPUTS_DIR / "routeA_phase1b_family_metrics.csv", index=False)
    confirmation.to_csv(OUTPUTS_DIR / "routeA_phase1b_dsunknown_confirmation.csv", index=False)

    write_phase1b_report(summary_metrics, confirmation, decision)
    write_phase1b_decision(decision)


if __name__ == "__main__":
    main()
