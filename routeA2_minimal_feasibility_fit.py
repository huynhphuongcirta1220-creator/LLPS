from __future__ import annotations

from pathlib import Path
import sys

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

if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    ROOT = Path.cwd()
    if not (ROOT / "outputs").exists() and (ROOT.parent / "outputs").exists():
        ROOT = ROOT.parent

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import routeA2_feature_audit_phase1 as a2_audit

FEATURE_SET_VERSION = "routeA2_explicit_transition_phase1"
BASELINE_FEATURE_SET_VERSION = "legacy_aligned_backbone_v1"
BASELINE_MODEL_NAME = "logreg"
RESIDUAL_MODEL_NAME = "ridge_linear_residual_routeA2_minimal"
PRIMARY_PROTOCOL = "strict_amine_object_holdout"
CONFIRM_PROTOCOL = "ds_unknown"
TARGET_SYSTEM_TYPE = "single_amine_solvent"
TARGET_FAMILIES = ["alcohol", "sulfur_polar_aprotic"]
PROB_CLIP_EPS = 1e-6
MIN_TRAIN_ROWS_FOR_LINEAR = 8
PRIMARY_SUCCESS_DELTA = 0.01
ALPHA_BY_FAMILY = {
    "alcohol": 1000.0,
    "sulfur_polar_aprotic": 2000.0,
}
FEATURE_COLUMNS = [
    "main_Am_acc_neu",
    "main_Am_don_neu",
    "main_Am_np_neu",
    "main_Am_wid_neu",
    "main_Am_acc_rea",
    "main_Am_don_rea",
    "main_Am_np_rea",
    "main_Am_wid_rea",
    "main_Delta_Am_acc",
    "main_Delta_Am_don",
    "main_Delta_Am_np",
    "main_Delta_Am_wid",
]

OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"
NOTEBOOKS_DIR = ROOT / "notebooks"


def load_old_routeA_predictions() -> pd.DataFrame:
    old_df = pd.read_csv(OUTPUTS_DIR / "routeA_phase1b_predictions.csv").copy()
    old_df = old_df.loc[
        old_df["protocol"].isin([PRIMARY_PROTOCOL, CONFIRM_PROTOCOL])
        & old_df["routeA_family"].isin(TARGET_FAMILIES)
        & (old_df["system_type"] == TARGET_SYSTEM_TYPE)
    ].copy()
    keep_cols = [
        "row_id",
        "protocol",
        "fold_id",
        "routeA_family",
        "y_true",
        "p_logreg_baseline",
        "p_routeA_corrected",
        "fit_mode",
    ]
    return old_df[keep_cols].rename(
        columns={
            "p_routeA_corrected": "p_old_routeA_phase1b",
            "fit_mode": "old_routeA_fit_mode",
        }
    )


def build_routeA2_feature_table() -> pd.DataFrame:
    mapping = a2_audit.build_canonical_map()
    descriptor_df = a2_audit.build_descriptor_table(mapping)
    alignment_df = a2_audit.build_alignment_audit(descriptor_df)
    route_df = pd.read_csv(OUTPUTS_DIR / "routeA_family_aware_increment_ready_table.csv").copy()
    route_df = route_df.loc[
        route_df["protocol"].isin([PRIMARY_PROTOCOL, CONFIRM_PROTOCOL])
        & (route_df["system_type"] == TARGET_SYSTEM_TYPE)
    ].copy()
    route_df["routeA2_family"] = route_df["solvent_family"].map(a2_audit.ROUTEA_FAMILY_MAP).fillna("other_organic_family")
    route_df = route_df.loc[route_df["routeA2_family"].isin(TARGET_FAMILIES)].copy()

    lookup = alignment_df.set_index("molecule_name").to_dict(orient="index")
    rows: list[dict[str, object]] = []
    for row in route_df.itertuples(index=False):
        amines = [str(row.amine1_name_canonical).strip()]
        if isinstance(row.amine2_name_canonical, str) and row.amine2_name_canonical.strip():
            amines.append(row.amine2_name_canonical.strip())
        records = [lookup.get(name, {}) for name in amines]
        main = records[0] if records else {}
        rows.append(
            {
                "feature_set_version": FEATURE_SET_VERSION,
                "protocol": row.protocol,
                "fold_id": int(row.fold_id),
                "routeA_family": row.routeA2_family,
                "system_type": row.system_type,
                "row_id": row.row_id,
                "amine1_name_canonical": row.amine1_name_canonical,
                "amine2_name_canonical": row.amine2_name_canonical,
                "solvent_name_canonical": row.solvent_name_canonical,
                "y_true": int(row.y_true),
                "p_logreg_baseline": float(row.p_logreg),
                "baseline_residual": float(row.residual),
                "main_aligned_object_name": amines[0] if amines else "",
                "main_aligned_form": str(main.get("main_aligned_form", "none")),
                "main_reactive_role": str(main.get("reactive_role", "")),
                "main_reacted_channel_main": str(main.get("reacted_channel_main", "")),
                "main_alignment_ready": int(main.get("main_alignment_ready", 0)),
                "explicit_transition_ready": int(main.get("explicit_transition_ready", 0)),
                "row_any_uncovered_reactive": int(
                    any(
                        str(rec.get("reactive_role", "")) == "reactive_amine"
                        and int(rec.get("main_alignment_ready", 0)) == 0
                        for rec in records
                    )
                ),
            }
        )
        for col in FEATURE_COLUMNS:
            rows[-1][col] = main.get(col, np.nan)

    feature_df = pd.DataFrame(rows)
    feature_df = feature_df.loc[feature_df["explicit_transition_ready"] == 1].copy()
    return feature_df


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
    if len(y_true) and y_true.nunique() >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["average_precision"] = float(average_precision_score(y_true, y_score))
    if len(y_true):
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


def fit_linear_residual_model(x_train: pd.DataFrame, y_train: pd.Series, family: str) -> tuple[Pipeline | None, str]:
    if len(x_train) < MIN_TRAIN_ROWS_FOR_LINEAR:
        return None, "intercept_only_low_support"
    alpha = ALPHA_BY_FAMILY[family]
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


def run_family_protocol(feature_df: pd.DataFrame, family: str, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = feature_df.loc[
        (feature_df["routeA_family"] == family)
        & (feature_df["protocol"] == protocol)
    ].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    prediction_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    for fold_id in sorted(int(x) for x in subset["fold_id"].unique()):
        train_df = subset.loc[subset["fold_id"] != fold_id].copy()
        test_df = subset.loc[subset["fold_id"] == fold_id].copy()
        x_train = train_df[FEATURE_COLUMNS].copy()
        y_train = train_df["baseline_residual"].astype(float)
        x_test = test_df[FEATURE_COLUMNS].copy()
        model, fit_mode = fit_linear_residual_model(x_train, y_train, family)

        if model is None:
            residual_hat_raw = np.repeat(float(y_train.mean()) if len(y_train) else 0.0, len(test_df))
            intercept_value = float(y_train.mean()) if len(y_train) else 0.0
            coef_map = {col: 0.0 for col in FEATURE_COLUMNS}
        else:
            residual_hat_raw = model.predict(x_test)
            ridge = model.named_steps["ridge"]
            intercept_value = float(ridge.intercept_)
            coef_map = dict(zip(FEATURE_COLUMNS, ridge.coef_.tolist()))

        base_prob = test_df["p_logreg_baseline"].astype(float).to_numpy()
        residual_hat_raw = np.asarray(residual_hat_raw, dtype=float)
        residual_hat, corrected_prob, clipped_flag = feasible_corrected_prob(base_prob, residual_hat_raw)

        for idx, row in enumerate(test_df.itertuples(index=False)):
            prediction_rows.append(
                {
                    "row_id": row.row_id,
                    "protocol": protocol,
                    "fold_id": int(fold_id),
                    "routeA_family": family,
                    "system_type": row.system_type,
                    "solvent_name_canonical": row.solvent_name_canonical,
                    "amine1_name_canonical": row.amine1_name_canonical,
                    "amine2_name_canonical": row.amine2_name_canonical,
                    "feature_set_version": FEATURE_SET_VERSION,
                    "baseline_feature_set_version": BASELINE_FEATURE_SET_VERSION,
                    "baseline_model_name": BASELINE_MODEL_NAME,
                    "residual_model_name": RESIDUAL_MODEL_NAME,
                    "y_true": int(row.y_true),
                    "p_logreg_baseline": float(row.p_logreg_baseline),
                    "baseline_residual": float(row.baseline_residual),
                    "residual_target": float(row.baseline_residual),
                    "residual_hat_raw": float(residual_hat_raw[idx]),
                    "residual_hat": float(residual_hat[idx]),
                    "residual_hat_feasible_clip_flag": int(clipped_flag[idx]),
                    "p_routeA2_corrected": float(corrected_prob[idx]),
                    "routeA2_corrected_residual": float(int(row.y_true) - corrected_prob[idx]),
                    "fit_mode": fit_mode,
                    "train_rows_used": int(len(train_df)),
                    "test_rows_used": int(len(test_df)),
                }
            )

        coef_row: dict[str, object] = {
            "protocol": protocol,
            "fold_id": int(fold_id),
            "routeA_family": family,
            "fit_mode": fit_mode,
            "intercept": intercept_value,
            "train_rows_used": int(len(train_df)),
            "test_rows_used": int(len(test_df)),
        }
        coef_row.update({col: float(coef_map[col]) for col in FEATURE_COLUMNS})
        coef_rows.append(coef_row)

    return pd.DataFrame(prediction_rows), pd.DataFrame(coef_rows)


def build_prediction_table(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_frames: list[pd.DataFrame] = []
    coef_frames: list[pd.DataFrame] = []
    for family in TARGET_FAMILIES:
        for protocol in [PRIMARY_PROTOCOL, CONFIRM_PROTOCOL]:
            pred_df, coef_df = run_family_protocol(feature_df, family, protocol)
            if not pred_df.empty:
                pred_frames.append(pred_df)
            if not coef_df.empty:
                coef_frames.append(coef_df)
    predictions = pd.concat(pred_frames, ignore_index=True)
    coefficients = pd.concat(coef_frames, ignore_index=True)
    old_df = load_old_routeA_predictions()
    merged = predictions.merge(
        old_df,
        on=["row_id", "protocol", "fold_id", "routeA_family", "y_true", "p_logreg_baseline"],
        how="left",
        validate="one_to_one",
    )
    return merged, coefficients


def metric_delta(corrected: float, baseline: float) -> float:
    if pd.isna(corrected) or pd.isna(baseline):
        return np.nan
    return float(corrected - baseline)


def subset_metrics(pred_df: pd.DataFrame, family: str, protocol: str, variant: str, fold_id: int | None = None) -> dict[str, object]:
    subset = pred_df.loc[
        (pred_df["routeA_family"] == family)
        & (pred_df["protocol"] == protocol)
    ].copy()
    if fold_id is not None:
        subset = subset.loc[subset["fold_id"] == fold_id].copy()
    score_col = {
        "baseline_only": "p_logreg_baseline",
        "old_routeA_phase1b_best_attempt": "p_old_routeA_phase1b",
        "routeA2_minimal_feasibility_fit": "p_routeA2_corrected",
    }[variant]
    metrics = compute_metrics(subset["y_true"], subset[score_col])
    metrics.update(
        {
            "feature_set_version": FEATURE_SET_VERSION,
            "baseline_feature_set_version": BASELINE_FEATURE_SET_VERSION,
            "baseline_model_name": BASELINE_MODEL_NAME,
            "residual_model_name": RESIDUAL_MODEL_NAME if variant == "routeA2_minimal_feasibility_fit" else "",
            "protocol": protocol,
            "routeA_family": family,
            "variant": variant,
            "fold_id": "all" if fold_id is None else int(fold_id),
        }
    )
    return metrics


def build_family_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in TARGET_FAMILIES:
        for protocol in [PRIMARY_PROTOCOL, CONFIRM_PROTOCOL]:
            subset = pred_df.loc[
                (pred_df["routeA_family"] == family)
                & (pred_df["protocol"] == protocol)
            ].copy()
            if subset.empty:
                continue
            for fold_id in sorted(int(x) for x in subset["fold_id"].unique()):
                for variant in [
                    "baseline_only",
                    "old_routeA_phase1b_best_attempt",
                    "routeA2_minimal_feasibility_fit",
                ]:
                    rows.append(subset_metrics(subset, family, protocol, variant, fold_id))
    return pd.DataFrame(rows)


def summarize_coefficients(coef_df: pd.DataFrame, family: str, protocol: str) -> dict[str, object]:
    subset = coef_df.loc[
        (coef_df["routeA_family"] == family)
        & (coef_df["protocol"] == protocol)
    ].copy()
    if subset.empty:
        return {}
    ranking = []
    for col in FEATURE_COLUMNS:
        ranking.append((col, float(subset[col].mean()), float(subset[col].abs().mean())))
    ranking.sort(key=lambda x: x[2], reverse=True)
    out: dict[str, object] = {}
    for idx, (name, mean_coef, mean_abs_coef) in enumerate(ranking[:3], start=1):
        out[f"top_feature_{idx}"] = name
        out[f"top_feature_{idx}_mean_coef"] = mean_coef
        out[f"top_feature_{idx}_mean_abs_coef"] = mean_abs_coef
    return out


def classify_improvement(delta_roc: float, delta_ap: float, delta_error: float, delta_brier: float, delta_log_loss: float) -> str:
    ranking_gain = (
        (pd.notna(delta_roc) and delta_roc >= PRIMARY_SUCCESS_DELTA)
        or (pd.notna(delta_ap) and delta_ap >= PRIMARY_SUCCESS_DELTA)
    )
    calibration_gain = (
        (pd.notna(delta_error) and delta_error < 0.0)
        or (pd.notna(delta_brier) and delta_brier < 0.0)
        or (pd.notna(delta_log_loss) and delta_log_loss < 0.0)
    )
    if ranking_gain and calibration_gain:
        return "ranking_plus_calibration"
    if ranking_gain:
        return "ranking_only"
    if calibration_gain:
        return "calibration_or_threshold_only"
    return "no_gain_or_degradation"


def build_summary_metrics(pred_df: pd.DataFrame, coef_df: pd.DataFrame, family_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in TARGET_FAMILIES:
        for protocol in [PRIMARY_PROTOCOL, CONFIRM_PROTOCOL]:
            subset = pred_df.loc[
                (pred_df["routeA_family"] == family)
                & (pred_df["protocol"] == protocol)
            ].copy()
            if subset.empty:
                continue
            base = subset_metrics(subset, family, protocol, "baseline_only")
            old = subset_metrics(subset, family, protocol, "old_routeA_phase1b_best_attempt")
            a2 = subset_metrics(subset, family, protocol, "routeA2_minimal_feasibility_fit")
            fold_subset = family_metrics.loc[
                (family_metrics["routeA_family"] == family)
                & (family_metrics["protocol"] == protocol)
            ].copy()
            for label, metrics in [("baseline_only", base), ("old_routeA_phase1b_best_attempt", old), ("routeA2_minimal_feasibility_fit", a2)]:
                label_folds = fold_subset.loc[fold_subset["variant"] == label].copy()
                row = {
                    "feature_set_version": FEATURE_SET_VERSION,
                    "baseline_feature_set_version": BASELINE_FEATURE_SET_VERSION,
                    "baseline_model_name": BASELINE_MODEL_NAME,
                    "protocol": protocol,
                    "routeA_family": family,
                    "variant": label,
                    **{k: metrics[k] for k in [
                        "support_rows", "n_pos", "n_neg", "positive_rate", "roc_auc", "average_precision",
                        "f1", "precision", "recall", "error_rate", "mean_residual", "mean_abs_residual",
                        "brier_score", "log_loss", "pred_mean", "calibration_drift", "abs_calibration_drift",
                    ]},
                    "fold_count": int(label_folds["fold_id"].nunique()),
                    "fold_mean_roc_auc": float(label_folds["roc_auc"].mean()),
                    "fold_std_roc_auc": float(label_folds["roc_auc"].std(ddof=1)) if len(label_folds) > 1 else np.nan,
                    "fold_mean_average_precision": float(label_folds["average_precision"].mean()),
                    "fold_std_average_precision": float(label_folds["average_precision"].std(ddof=1)) if len(label_folds) > 1 else np.nan,
                    "fold_mean_error_rate": float(label_folds["error_rate"].mean()),
                    "fold_std_error_rate": float(label_folds["error_rate"].std(ddof=1)) if len(label_folds) > 1 else np.nan,
                }
                if label == "routeA2_minimal_feasibility_fit":
                    row["pred_mean_shift_vs_baseline"] = metric_delta(a2["pred_mean"], base["pred_mean"])
                    row["delta_roc_auc_vs_baseline"] = metric_delta(a2["roc_auc"], base["roc_auc"])
                    row["delta_average_precision_vs_baseline"] = metric_delta(a2["average_precision"], base["average_precision"])
                    row["delta_error_rate_vs_baseline"] = metric_delta(a2["error_rate"], base["error_rate"])
                    row["delta_brier_score_vs_baseline"] = metric_delta(a2["brier_score"], base["brier_score"])
                    row["delta_log_loss_vs_baseline"] = metric_delta(a2["log_loss"], base["log_loss"])
                    row["delta_roc_auc_vs_old_routeA"] = metric_delta(a2["roc_auc"], old["roc_auc"])
                    row["delta_average_precision_vs_old_routeA"] = metric_delta(a2["average_precision"], old["average_precision"])
                    row["delta_error_rate_vs_old_routeA"] = metric_delta(a2["error_rate"], old["error_rate"])
                    row["improvement_mode_vs_baseline"] = classify_improvement(
                        row["delta_roc_auc_vs_baseline"],
                        row["delta_average_precision_vs_baseline"],
                        row["delta_error_rate_vs_baseline"],
                        row["delta_brier_score_vs_baseline"],
                        row["delta_log_loss_vs_baseline"],
                    )
                    row.update(summarize_coefficients(coef_df, family, protocol))
                rows.append(row)
    return pd.DataFrame(rows)


def build_dsunknown_confirmation(summary_df: pd.DataFrame) -> pd.DataFrame:
    base = summary_df.loc[summary_df["variant"] == "routeA2_minimal_feasibility_fit"].copy()
    strict = base.loc[base["protocol"] == PRIMARY_PROTOCOL].copy()
    strict = strict.rename(
        columns={
            "delta_roc_auc_vs_baseline": "strict_delta_roc_auc",
            "delta_average_precision_vs_baseline": "strict_delta_average_precision",
            "delta_error_rate_vs_baseline": "strict_delta_error_rate",
            "delta_brier_score_vs_baseline": "strict_delta_brier_score",
            "delta_log_loss_vs_baseline": "strict_delta_log_loss",
            "improvement_mode_vs_baseline": "strict_improvement_mode",
        }
    )
    strict = strict[[
        "routeA_family", "strict_delta_roc_auc", "strict_delta_average_precision",
        "strict_delta_error_rate", "strict_delta_brier_score", "strict_delta_log_loss",
        "strict_improvement_mode",
    ]]
    confirm = base.loc[base["protocol"] == CONFIRM_PROTOCOL].copy()
    confirm = confirm.merge(strict, on="routeA_family", how="left", validate="one_to_one")
    confirm["direction_consistent_with_strict"] = (
        (
            (confirm["strict_delta_roc_auc"].fillna(0.0) >= 0.0)
            & (confirm["delta_roc_auc_vs_baseline"].fillna(0.0) >= 0.0)
        ) | (
            (confirm["strict_delta_average_precision"].fillna(0.0) >= 0.0)
            & (confirm["delta_average_precision_vs_baseline"].fillna(0.0) >= 0.0)
        )
    ).astype(int)
    return confirm[[
        "feature_set_version", "baseline_feature_set_version", "baseline_model_name", "protocol", "routeA_family",
        "support_rows", "roc_auc", "average_precision", "error_rate", "brier_score", "log_loss",
        "delta_roc_auc_vs_baseline", "delta_average_precision_vs_baseline", "delta_error_rate_vs_baseline",
        "delta_brier_score_vs_baseline", "delta_log_loss_vs_baseline", "strict_delta_roc_auc",
        "strict_delta_average_precision", "strict_delta_error_rate", "strict_delta_brier_score",
        "strict_delta_log_loss", "direction_consistent_with_strict", "improvement_mode_vs_baseline",
        "strict_improvement_mode",
    ]].copy()


def build_decision_table(summary_df: pd.DataFrame, confirm_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    a2_df = summary_df.loc[summary_df["variant"] == "routeA2_minimal_feasibility_fit"].copy()
    for family in TARGET_FAMILIES:
        strict_row = a2_df.loc[(a2_df["protocol"] == PRIMARY_PROTOCOL) & (a2_df["routeA_family"] == family)].iloc[0]
        confirm_row = confirm_df.loc[confirm_df["routeA_family"] == family].iloc[0]
        ranking_ok = (
            (pd.notna(strict_row["delta_roc_auc_vs_baseline"]) and strict_row["delta_roc_auc_vs_baseline"] >= PRIMARY_SUCCESS_DELTA)
            or (pd.notna(strict_row["delta_average_precision_vs_baseline"]) and strict_row["delta_average_precision_vs_baseline"] >= PRIMARY_SUCCESS_DELTA)
        )
        error_ok = pd.notna(strict_row["delta_error_rate_vs_baseline"]) and strict_row["delta_error_rate_vs_baseline"] <= 1e-12
        calibration_ok = (
            pd.notna(strict_row["delta_brier_score_vs_baseline"])
            and pd.notna(strict_row["delta_log_loss_vs_baseline"])
            and strict_row["delta_brier_score_vs_baseline"] <= 1e-12
            and strict_row["delta_log_loss_vs_baseline"] <= 1e-12
        )
        direction_ok = int(confirm_row["direction_consistent_with_strict"]) == 1
        status = "established" if ranking_ok and error_ok and calibration_ok and direction_ok else "not_established"
        rows.append(
            {
                "routeA_family": family,
                "strict_support_rows": int(strict_row["support_rows"]),
                "strict_delta_roc_auc_vs_baseline": strict_row["delta_roc_auc_vs_baseline"],
                "strict_delta_average_precision_vs_baseline": strict_row["delta_average_precision_vs_baseline"],
                "strict_delta_error_rate_vs_baseline": strict_row["delta_error_rate_vs_baseline"],
                "strict_delta_brier_score_vs_baseline": strict_row["delta_brier_score_vs_baseline"],
                "strict_delta_log_loss_vs_baseline": strict_row["delta_log_loss_vs_baseline"],
                "strict_delta_roc_auc_vs_old_routeA": strict_row["delta_roc_auc_vs_old_routeA"],
                "strict_delta_average_precision_vs_old_routeA": strict_row["delta_average_precision_vs_old_routeA"],
                "strict_pred_mean_shift_vs_baseline": strict_row["pred_mean_shift_vs_baseline"],
                "strict_improvement_mode": strict_row["improvement_mode_vs_baseline"],
                "ds_unknown_delta_roc_auc_vs_baseline": confirm_row["delta_roc_auc_vs_baseline"],
                "ds_unknown_delta_average_precision_vs_baseline": confirm_row["delta_average_precision_vs_baseline"],
                "ds_unknown_delta_error_rate_vs_baseline": confirm_row["delta_error_rate_vs_baseline"],
                "ds_unknown_delta_brier_score_vs_baseline": confirm_row["delta_brier_score_vs_baseline"],
                "ds_unknown_delta_log_loss_vs_baseline": confirm_row["delta_log_loss_vs_baseline"],
                "ds_unknown_direction_consistent": int(confirm_row["direction_consistent_with_strict"]),
                "top_feature_1": strict_row.get("top_feature_1", ""),
                "top_feature_1_mean_coef": strict_row.get("top_feature_1_mean_coef", np.nan),
                "top_feature_2": strict_row.get("top_feature_2", ""),
                "top_feature_2_mean_coef": strict_row.get("top_feature_2_mean_coef", np.nan),
                "top_feature_3": strict_row.get("top_feature_3", ""),
                "top_feature_3_mean_coef": strict_row.get("top_feature_3_mean_coef", np.nan),
                "routeA2_minimal_status": status,
            }
        )
    return pd.DataFrame(rows)


def write_reports(summary_df: pd.DataFrame, confirm_df: pd.DataFrame, decision_df: pd.DataFrame) -> None:
    strict_a2 = summary_df.loc[
        (summary_df["protocol"] == PRIMARY_PROTOCOL)
        & (summary_df["variant"] == "routeA2_minimal_feasibility_fit")
    ].copy()
    strict_old = summary_df.loc[
        (summary_df["protocol"] == PRIMARY_PROTOCOL)
        & (summary_df["variant"] == "old_routeA_phase1b_best_attempt")
    ].copy()

    lines = [
        "# Route A2 Minimal Feasibility Fit",
        "",
        "## Scope",
        "",
        "- Official baseline remains frozen.",
        "- Residual target remains `r = y_true - p_logreg`.",
        "- Only 12 main-channel explicit aligned features are used.",
        "- Main test: `strict + single_amine_solvent + alcohol`.",
        "- Secondary test: `strict + single_amine_solvent + sulfur_polar_aprotic`.",
        "- `DS-unknown` is confirmation only.",
        "",
        "## Why This Is Not Old Route A",
        "",
        "- Old Route A phase1b used summary-compressed reacted deltas.",
        "- Route A2 uses explicit aligned `Am_*_neu`, `Am_*_rea`, and `Delta_Am_*` on the same segmentation axis.",
        "",
        "## Strict Comparison",
        "",
    ]
    for family in TARGET_FAMILIES:
        base = summary_df.loc[
            (summary_df["protocol"] == PRIMARY_PROTOCOL)
            & (summary_df["routeA_family"] == family)
            & (summary_df["variant"] == "baseline_only")
        ].iloc[0]
        old = strict_old.loc[strict_old["routeA_family"] == family].iloc[0]
        a2 = strict_a2.loc[strict_a2["routeA_family"] == family].iloc[0]
        lines.append(
            f"- {family}: baseline ROC/AP {base['roc_auc']:.3f}/{base['average_precision']:.3f}; "
            f"old Route A {old['roc_auc']:.3f}/{old['average_precision']:.3f}; "
            f"Route A2 {a2['roc_auc']:.3f}/{a2['average_precision']:.3f}. "
            f"Route A2 deltas vs baseline: ROC {a2['delta_roc_auc_vs_baseline']:+.3f}, AP {a2['delta_average_precision_vs_baseline']:+.3f}, "
            f"error {a2['delta_error_rate_vs_baseline']:+.3f}, Brier {a2['delta_brier_score_vs_baseline']:+.3f}, "
            f"log loss {a2['delta_log_loss_vs_baseline']:+.3f}, mode `{a2['improvement_mode_vs_baseline']}`."
        )

    lines.extend(["", "## DS-unknown Confirmation", ""])
    for row in confirm_df.itertuples(index=False):
        lines.append(
            f"- {row.routeA_family}: ROC delta {row.delta_roc_auc_vs_baseline:+.3f}, AP delta {row.delta_average_precision_vs_baseline:+.3f}, "
            f"error delta {row.delta_error_rate_vs_baseline:+.3f}, Brier delta {row.delta_brier_score_vs_baseline:+.3f}, "
            f"log-loss delta {row.delta_log_loss_vs_baseline:+.3f}, direction-consistent={int(row.direction_consistent_with_strict)}."
        )
    lines.extend(["", "## Coefficient Signal", ""])
    for row in decision_df.itertuples(index=False):
        lines.append(
            f"- {row.routeA_family}: `{row.top_feature_1}` ({row.top_feature_1_mean_coef:+.3f}), "
            f"`{row.top_feature_2}` ({row.top_feature_2_mean_coef:+.3f}), "
            f"`{row.top_feature_3}` ({row.top_feature_3_mean_coef:+.3f}) -> `{row.routeA2_minimal_status}`."
        )
    (REPORTS_DIR / "routeA2_minimal_fit.md").write_text("\n".join(lines), encoding="utf-8")

    overall = "A2_not_established_after_minimal_fit"
    if (decision_df["routeA2_minimal_status"] == "established").any():
        overall = "continue_cautiously"
    lines = [
        "# Route A2 Minimal Fit Decision",
        "",
        f"- alcohol: `{decision_df.loc[decision_df['routeA_family'] == 'alcohol', 'routeA2_minimal_status'].iloc[0]}`",
        f"- sulfur_polar_aprotic: `{decision_df.loc[decision_df['routeA_family'] == 'sulfur_polar_aprotic', 'routeA2_minimal_status'].iloc[0]}`",
        "",
        f"- Overall: `{overall}`",
        "- If not established, Route A2 should remain exploratory and should not expand family or feature scope.",
    ]
    (REPORTS_DIR / "routeA2_minimal_fit_decision.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    feature_df = build_routeA2_feature_table()
    pred_df, coef_df = build_prediction_table(feature_df)
    family_metrics = build_family_metrics(pred_df)
    summary_metrics = build_summary_metrics(pred_df, coef_df, family_metrics)
    confirm_df = build_dsunknown_confirmation(summary_metrics)
    decision_df = build_decision_table(summary_metrics, confirm_df)

    pred_df.to_csv(OUTPUTS_DIR / "routeA2_minimal_fit_predictions.csv", index=False)
    summary_metrics.to_csv(OUTPUTS_DIR / "routeA2_minimal_fit_metrics.csv", index=False)
    family_metrics.to_csv(OUTPUTS_DIR / "routeA2_minimal_fit_family_metrics.csv", index=False)
    confirm_df.to_csv(OUTPUTS_DIR / "routeA2_minimal_fit_dsunknown_confirmation.csv", index=False)
    write_reports(summary_metrics, confirm_df, decision_df)


if __name__ == "__main__":
    main()
