from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

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

FEATURE_SET_VERSION = "routeA2_explicit_transition_capacity_stress"
BASELINE_FEATURE_SET_VERSION = "legacy_aligned_backbone_v1"
BASELINE_MODEL_NAME = "logreg"
PRIMARY_PROTOCOL = "strict_amine_object_holdout"
CONFIRM_PROTOCOL = "ds_unknown"
TARGET_SYSTEM_TYPE = "single_amine_solvent"
TARGET_FAMILIES = ["alcohol", "sulfur_polar_aprotic", "glycol_ether_glyme"]
PROB_CLIP_EPS = 1e-6
MIN_TRAIN_ROWS = 8
PRIMARY_SUCCESS_DELTA = 0.01
MODEL_ORDER = ["svm_rbf", "random_forest", "xgboost", "lightgbm"]
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

MODEL_PARAM_TEXT = {
    "svm_rbf": "SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.05)",
    "random_forest": "RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=3, random_state=20260417)",
    "xgboost": "XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:squarederror', tree_method='hist', random_state=20260417, n_jobs=1)",
    "lightgbm": "LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, num_leaves=15, subsample=0.8, colsample_bytree=0.8, min_child_samples=10, reg_lambda=0.0, random_state=20260417, n_jobs=1, verbosity=-1)",
}


def load_old_routeA_predictions() -> pd.DataFrame:
    path = OUTPUTS_DIR / "routeA_phase1b_predictions.csv"
    if not path.exists():
        return pd.DataFrame(columns=["row_id", "protocol", "fold_id", "routeA_family", "p_old_routeA_phase1b"])
    old_df = pd.read_csv(path).copy()
    old_df = old_df.loc[
        old_df["protocol"].isin([PRIMARY_PROTOCOL, CONFIRM_PROTOCOL])
        & old_df["routeA_family"].isin(TARGET_FAMILIES)
        & (old_df["system_type"] == TARGET_SYSTEM_TYPE)
    ].copy()
    keep_cols = ["row_id", "protocol", "fold_id", "routeA_family", "p_routeA_corrected"]
    return old_df[keep_cols].rename(columns={"p_routeA_corrected": "p_old_routeA_phase1b"})


def build_routeA2_feature_table() -> pd.DataFrame:
    mapping = a2_audit.build_canonical_map()
    descriptor_df = a2_audit.build_descriptor_table(mapping)
    alignment_df = a2_audit.build_alignment_audit(descriptor_df)
    route_df = pd.read_csv(OUTPUTS_DIR / "routeA_family_aware_increment_ready_table.csv").copy()
    route_df = route_df.loc[
        route_df["protocol"].isin([PRIMARY_PROTOCOL, CONFIRM_PROTOCOL])
        & (route_df["system_type"] == TARGET_SYSTEM_TYPE)
    ].copy()
    route_df["routeA_family"] = route_df["solvent_family"].map(a2_audit.ROUTEA_FAMILY_MAP).fillna("other_organic_family")
    route_df = route_df.loc[route_df["routeA_family"].isin(TARGET_FAMILIES)].copy()

    lookup = alignment_df.set_index("molecule_name").to_dict(orient="index")
    rows: list[dict[str, object]] = []
    for row in route_df.itertuples(index=False):
        amines = [str(row.amine1_name_canonical).strip()]
        if isinstance(row.amine2_name_canonical, str) and row.amine2_name_canonical.strip():
            amines.append(row.amine2_name_canonical.strip())
        records = [lookup.get(name, {}) for name in amines]
        main = records[0] if records else {}
        record = {
            "feature_set_version": FEATURE_SET_VERSION,
            "protocol": row.protocol,
            "fold_id": int(row.fold_id),
            "routeA_family": row.routeA_family,
            "system_type": row.system_type,
            "row_id": row.row_id,
            "amine1_name_canonical": row.amine1_name_canonical,
            "amine2_name_canonical": row.amine2_name_canonical,
            "solvent_name_canonical": row.solvent_name_canonical,
            "y_true": int(row.y_true),
            "p_logreg_baseline": float(row.p_logreg),
            "baseline_residual": float(row.residual),
            "main_alignment_ready": int(main.get("main_alignment_ready", 0)),
            "explicit_transition_ready": int(main.get("explicit_transition_ready", 0)),
        }
        for col in FEATURE_COLUMNS:
            record[col] = main.get(col, np.nan)
        rows.append(record)
    feature_df = pd.DataFrame(rows)
    return feature_df.loc[feature_df["explicit_transition_ready"] == 1].copy()


def make_model(model_name: str):
    if model_name == "svm_rbf":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.05)),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("model", RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=3, random_state=20260417)),
            ]
        )
    if model_name == "xgboost":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("model", XGBRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=20260417,
                    n_jobs=1,
                )),
            ]
        )
    if model_name == "lightgbm":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("model", LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    num_leaves=15,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=10,
                    reg_lambda=0.0,
                    random_state=20260417,
                    n_jobs=1,
                    verbosity=-1,
                )),
            ]
        )
    raise ValueError(model_name)


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
    }
    if len(y_true) and y_true.nunique() >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["average_precision"] = float(average_precision_score(y_true, y_score))
    if len(y_true):
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        out["f1"] = float(f1)
        out["precision"] = float(precision)
        out["recall"] = float(recall)
    return out


def feasible_corrected_prob(base_prob: np.ndarray, raw_residual_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = -base_prob
    upper = 1.0 - base_prob
    feasible_residual_hat = np.clip(raw_residual_hat, lower, upper)
    corrected_prob = np.clip(base_prob + feasible_residual_hat, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)
    clipped_flag = (np.abs(feasible_residual_hat - raw_residual_hat) > 1e-12).astype(int)
    return feasible_residual_hat, corrected_prob, clipped_flag


def run_family_protocol(feature_df: pd.DataFrame, family: str, protocol: str, model_name: str) -> pd.DataFrame:
    subset = feature_df.loc[
        (feature_df["routeA_family"] == family)
        & (feature_df["protocol"] == protocol)
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for fold_id in sorted(int(x) for x in subset["fold_id"].unique()):
        train_df = subset.loc[subset["fold_id"] != fold_id].copy()
        test_df = subset.loc[subset["fold_id"] == fold_id].copy()
        if len(train_df) < MIN_TRAIN_ROWS:
            residual_hat_raw = np.repeat(float(train_df["baseline_residual"].mean()) if len(train_df) else 0.0, len(test_df))
            fit_mode = "intercept_only_low_support"
        else:
            model = make_model(model_name)
            model.fit(train_df[FEATURE_COLUMNS], train_df["baseline_residual"].astype(float))
            residual_hat_raw = model.predict(test_df[FEATURE_COLUMNS])
            fit_mode = MODEL_PARAM_TEXT[model_name]
        base_prob = test_df["p_logreg_baseline"].astype(float).to_numpy()
        residual_hat_raw = np.asarray(residual_hat_raw, dtype=float)
        residual_hat, corrected_prob, clipped_flag = feasible_corrected_prob(base_prob, residual_hat_raw)
        for idx, row in enumerate(test_df.itertuples(index=False)):
            rows.append(
                {
                    "row_id": row.row_id,
                    "protocol": protocol,
                    "fold_id": int(fold_id),
                    "routeA_family": family,
                    "model_name": model_name,
                    "system_type": row.system_type,
                    "amine1_name_canonical": row.amine1_name_canonical,
                    "amine2_name_canonical": row.amine2_name_canonical,
                    "solvent_name_canonical": row.solvent_name_canonical,
                    "y_true": int(row.y_true),
                    "p_logreg_baseline": float(row.p_logreg_baseline),
                    "baseline_residual": float(row.baseline_residual),
                    "residual_hat_raw": float(residual_hat_raw[idx]),
                    "residual_hat": float(residual_hat[idx]),
                    "residual_hat_feasible_clip_flag": int(clipped_flag[idx]),
                    "p_routeA2_corrected": float(corrected_prob[idx]),
                    "fit_mode": fit_mode,
                    "train_rows_used": int(len(train_df)),
                    "test_rows_used": int(len(test_df)),
                }
            )
    return pd.DataFrame(rows)


def build_predictions(feature_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for family in TARGET_FAMILIES:
        for protocol in [PRIMARY_PROTOCOL, CONFIRM_PROTOCOL]:
            for model_name in MODEL_ORDER:
                pred_df = run_family_protocol(feature_df, family, protocol, model_name)
                if not pred_df.empty:
                    frames.append(pred_df)
    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions.merge(
        load_old_routeA_predictions(),
        on=["row_id", "protocol", "fold_id", "routeA_family"],
        how="left",
        validate="many_to_one",
    )
    predictions["feature_set_version"] = FEATURE_SET_VERSION
    predictions["baseline_feature_set_version"] = BASELINE_FEATURE_SET_VERSION
    return predictions


def subset_metric_row(pred_df: pd.DataFrame, family: str, protocol: str, model_name: str, variant: str, fold_id: int | None = None) -> dict[str, object]:
    subset = pred_df.loc[
        (pred_df["routeA_family"] == family)
        & (pred_df["protocol"] == protocol)
        & (pred_df["model_name"] == model_name)
    ].copy()
    if fold_id is not None:
        subset = subset.loc[subset["fold_id"] == fold_id].copy()
    score_col = {
        "baseline_only": "p_logreg_baseline",
        "old_routeA_phase1b_best_attempt": "p_old_routeA_phase1b",
        "routeA2_capacity_stress": "p_routeA2_corrected",
    }[variant]
    metrics = compute_metrics(subset["y_true"], subset[score_col])
    metrics.update(
        {
            "feature_set_version": FEATURE_SET_VERSION,
            "protocol": protocol,
            "routeA_family": family,
            "model_name": model_name,
            "variant": variant,
            "fold_id": "all" if fold_id is None else int(fold_id),
        }
    )
    return metrics


def build_family_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in TARGET_FAMILIES:
        for protocol in [PRIMARY_PROTOCOL, CONFIRM_PROTOCOL]:
            for model_name in MODEL_ORDER:
                subset = pred_df.loc[
                    (pred_df["routeA_family"] == family)
                    & (pred_df["protocol"] == protocol)
                    & (pred_df["model_name"] == model_name)
                ].copy()
                if subset.empty:
                    continue
                variants = ["baseline_only", "routeA2_capacity_stress"]
                if subset["p_old_routeA_phase1b"].notna().any():
                    variants.insert(1, "old_routeA_phase1b_best_attempt")
                for fold_id in sorted(int(x) for x in subset["fold_id"].unique()):
                    for variant in variants:
                        rows.append(subset_metric_row(subset, family, protocol, model_name, variant, fold_id))
    return pd.DataFrame(rows)


def metric_delta(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a - b)


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


def build_summary_metrics(pred_df: pd.DataFrame, family_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in TARGET_FAMILIES:
        for protocol in [PRIMARY_PROTOCOL, CONFIRM_PROTOCOL]:
            for model_name in MODEL_ORDER:
                subset = pred_df.loc[
                    (pred_df["routeA_family"] == family)
                    & (pred_df["protocol"] == protocol)
                    & (pred_df["model_name"] == model_name)
                ].copy()
                if subset.empty:
                    continue
                base = subset_metric_row(subset, family, protocol, model_name, "baseline_only")
                stress = subset_metric_row(subset, family, protocol, model_name, "routeA2_capacity_stress")
                old = subset_metric_row(subset, family, protocol, model_name, "old_routeA_phase1b_best_attempt") if subset["p_old_routeA_phase1b"].notna().any() else None
                fold_subset = family_metrics.loc[
                    (family_metrics["routeA_family"] == family)
                    & (family_metrics["protocol"] == protocol)
                    & (family_metrics["model_name"] == model_name)
                    & (family_metrics["variant"] == "routeA2_capacity_stress")
                ].copy()
                row = {
                    "feature_set_version": FEATURE_SET_VERSION,
                    "baseline_feature_set_version": BASELINE_FEATURE_SET_VERSION,
                    "protocol": protocol,
                    "routeA_family": family,
                    "model_name": model_name,
                    "support_rows": int(stress["support_rows"]),
                    "baseline_roc_auc": base["roc_auc"],
                    "stress_roc_auc": stress["roc_auc"],
                    "delta_roc_auc_vs_baseline": metric_delta(stress["roc_auc"], base["roc_auc"]),
                    "baseline_average_precision": base["average_precision"],
                    "stress_average_precision": stress["average_precision"],
                    "delta_average_precision_vs_baseline": metric_delta(stress["average_precision"], base["average_precision"]),
                    "baseline_f1": base["f1"],
                    "stress_f1": stress["f1"],
                    "baseline_precision": base["precision"],
                    "stress_precision": stress["precision"],
                    "baseline_recall": base["recall"],
                    "stress_recall": stress["recall"],
                    "baseline_error_rate": base["error_rate"],
                    "stress_error_rate": stress["error_rate"],
                    "delta_error_rate_vs_baseline": metric_delta(stress["error_rate"], base["error_rate"]),
                    "baseline_brier_score": base["brier_score"],
                    "stress_brier_score": stress["brier_score"],
                    "delta_brier_score_vs_baseline": metric_delta(stress["brier_score"], base["brier_score"]),
                    "baseline_log_loss": base["log_loss"],
                    "stress_log_loss": stress["log_loss"],
                    "delta_log_loss_vs_baseline": metric_delta(stress["log_loss"], base["log_loss"]),
                    "baseline_pred_mean": base["pred_mean"],
                    "stress_pred_mean": stress["pred_mean"],
                    "pred_mean_shift_vs_baseline": metric_delta(stress["pred_mean"], base["pred_mean"]),
                    "calibration_drift_baseline": base["calibration_drift"],
                    "calibration_drift_stress": stress["calibration_drift"],
                    "fold_count": int(fold_subset["fold_id"].nunique()),
                    "fold_mean_roc_auc": float(fold_subset["roc_auc"].mean()),
                    "fold_std_roc_auc": float(fold_subset["roc_auc"].std(ddof=1)) if len(fold_subset) > 1 else np.nan,
                    "fold_mean_average_precision": float(fold_subset["average_precision"].mean()),
                    "fold_std_average_precision": float(fold_subset["average_precision"].std(ddof=1)) if len(fold_subset) > 1 else np.nan,
                    "improvement_mode_vs_baseline": classify_improvement(
                        metric_delta(stress["roc_auc"], base["roc_auc"]),
                        metric_delta(stress["average_precision"], base["average_precision"]),
                        metric_delta(stress["error_rate"], base["error_rate"]),
                        metric_delta(stress["brier_score"], base["brier_score"]),
                        metric_delta(stress["log_loss"], base["log_loss"]),
                    ),
                    "fit_mode": subset["fit_mode"].iloc[0],
                }
                if old is not None:
                    row["old_routeA_roc_auc"] = old["roc_auc"]
                    row["old_routeA_average_precision"] = old["average_precision"]
                    row["delta_roc_auc_vs_old_routeA"] = metric_delta(stress["roc_auc"], old["roc_auc"])
                    row["delta_average_precision_vs_old_routeA"] = metric_delta(stress["average_precision"], old["average_precision"])
                rows.append(row)
    return pd.DataFrame(rows)


def build_dsunknown_confirmation(summary_df: pd.DataFrame) -> pd.DataFrame:
    strict = summary_df.loc[summary_df["protocol"] == PRIMARY_PROTOCOL].copy()
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
        "routeA_family", "model_name", "strict_delta_roc_auc", "strict_delta_average_precision",
        "strict_delta_error_rate", "strict_delta_brier_score", "strict_delta_log_loss", "strict_improvement_mode",
    ]]
    confirm = summary_df.loc[summary_df["protocol"] == CONFIRM_PROTOCOL].copy()
    confirm = confirm.merge(strict, on=["routeA_family", "model_name"], how="left", validate="one_to_one")
    confirm["direction_consistent_with_strict"] = (
        (
            (confirm["strict_delta_roc_auc"].fillna(0.0) >= 0.0)
            & (confirm["delta_roc_auc_vs_baseline"].fillna(0.0) >= 0.0)
        ) | (
            (confirm["strict_delta_average_precision"].fillna(0.0) >= 0.0)
            & (confirm["delta_average_precision_vs_baseline"].fillna(0.0) >= 0.0)
        )
    ).astype(int)
    return confirm.copy()


def write_reports(feature_df: pd.DataFrame, summary_df: pd.DataFrame, confirm_df: pd.DataFrame) -> None:
    coverage = (
        feature_df.groupby(["protocol", "routeA_family"], as_index=False)
        .agg(n_rows=("row_id", "count"), n_row_id=("row_id", "nunique"))
    )
    lines = [
        "# Route A2 Capacity Stress Test",
        "",
        "## Status",
        "",
        "- This is an exploratory stress-test line only.",
        "- It does not modify official baseline, official primary evidence, or the earlier `A2_not_established_after_minimal_fit` conclusion.",
        "- Purpose: test whether larger single-amine-solvent coverage plus higher-capacity residual learners can extract nonlinear signal from explicit aligned A2 transition features.",
        "",
        "## Scope",
        "",
        "- Protocols: `strict_amine_object_holdout` for training/evidence, `DS-unknown` for confirmation only.",
        "- System type: `single_amine_solvent` only.",
        "- Families: `alcohol`, `sulfur_polar_aprotic`, `glycol_ether_glyme`.",
        "- Features: only the 12 explicit aligned `Am_*_neu / Am_*_rea / Delta_Am_*` columns.",
        "- Models: `SVR-RBF`, `RandomForest`, `XGBoost`, `LightGBM` residual regressors.",
        "",
        "## Coverage",
        "",
    ]
    for row in coverage.itertuples(index=False):
        lines.append(f"- {row.protocol} / {row.routeA_family}: {row.n_rows} rows")
    lines.extend(["", "## Model Defaults", ""])
    for name in MODEL_ORDER:
        lines.append(f"- {name}: `{MODEL_PARAM_TEXT[name]}`")
    lines.extend(["", "## Strict Summary", ""])
    strict = summary_df.loc[summary_df["protocol"] == PRIMARY_PROTOCOL].copy()
    for family in TARGET_FAMILIES:
        fam = strict.loc[strict["routeA_family"] == family].sort_values(
            ["delta_roc_auc_vs_baseline", "delta_average_precision_vs_baseline"],
            ascending=[False, False],
        )
        if fam.empty:
            continue
        best = fam.iloc[0]
        lines.append(
            f"- {family}: best stress model `{best['model_name']}` with ROC delta {best['delta_roc_auc_vs_baseline']:+.3f}, "
            f"AP delta {best['delta_average_precision_vs_baseline']:+.3f}, error delta {best['delta_error_rate_vs_baseline']:+.3f}, "
            f"Brier delta {best['delta_brier_score_vs_baseline']:+.3f}, log-loss delta {best['delta_log_loss_vs_baseline']:+.3f}, "
            f"mode `{best['improvement_mode_vs_baseline']}`."
        )
    lines.extend(["", "## DS-unknown Confirmation", ""])
    for family in TARGET_FAMILIES:
        fam = confirm_df.loc[confirm_df["routeA_family"] == family].sort_values(
            ["delta_roc_auc_vs_baseline", "delta_average_precision_vs_baseline"],
            ascending=[False, False],
        )
        if fam.empty:
            continue
        best = fam.iloc[0]
        lines.append(
            f"- {family}: best DS-unknown stress model `{best['model_name']}` with ROC delta {best['delta_roc_auc_vs_baseline']:+.3f}, "
            f"AP delta {best['delta_average_precision_vs_baseline']:+.3f}, direction-consistent={int(best['direction_consistent_with_strict'])}."
        )
    (REPORTS_DIR / "routeA2_capacity_stress_test.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    feature_df = build_routeA2_feature_table()
    pred_df = build_predictions(feature_df)
    family_metrics = build_family_metrics(pred_df)
    summary_metrics = build_summary_metrics(pred_df, family_metrics)
    confirm_df = build_dsunknown_confirmation(summary_metrics)

    pred_df.to_csv(OUTPUTS_DIR / "routeA2_capacity_stress_test_predictions.csv", index=False)
    summary_metrics.to_csv(OUTPUTS_DIR / "routeA2_capacity_stress_test_metrics.csv", index=False)
    family_metrics.to_csv(OUTPUTS_DIR / "routeA2_capacity_stress_test_family_metrics.csv", index=False)
    confirm_df.to_csv(OUTPUTS_DIR / "routeA2_capacity_stress_test_dsunknown_confirmation.csv", index=False)
    write_reports(feature_df, summary_metrics, confirm_df)


if __name__ == "__main__":
    main()
