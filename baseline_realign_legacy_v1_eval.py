from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURE_SET_VERSION = "legacy_aligned_backbone_v1"
SEED = 20260417
PRIMARY_FOLD_OPTIONS = [5, 4, 3]
RANDOM_SEARCH_ITERS = 800

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"


@dataclass
class SplitAssignment:
    n_folds: int
    amine_to_fold: dict[str, int]
    fold_loads: list[int]
    eligible_rows: int
    mixed_rows: int
    score: float


def load_modeling_inputs() -> tuple[pd.DataFrame, list[str]]:
    modeling_df = pd.read_csv(OUTPUTS_DIR / "baseline_modeling_table_legacy_v1.csv")
    for col in ["amine1_name_canonical", "amine2_name_canonical", "solvent_name_canonical", "amine_set_key"]:
        modeling_df[col] = modeling_df[col].fillna("").astype(str)
    feature_df = pd.read_csv(OUTPUTS_DIR / "baseline_feature_dictionary_legacy_v1.csv")
    trainable_cols = feature_df.loc[feature_df["retained_as_trainable"] == 1, "column_name"].tolist()
    return modeling_df, trainable_cols


def assignment_score(df: pd.DataFrame, y: pd.Series, amine_to_fold: dict[str, int], n_folds: int) -> SplitAssignment:
    amine1 = df["amine1_name_canonical"].fillna("").to_numpy()
    amine2 = df["amine2_name_canonical"].fillna("").to_numpy()
    fold1 = np.array([amine_to_fold.get(a, -1) for a in amine1], dtype=int)
    fold2 = np.array([amine_to_fold.get(a, -1) for a in amine2], dtype=int)
    has_a2 = amine2 != ""
    eligible = (~has_a2) | (fold1 == fold2)
    mixed = has_a2 & (fold1 != fold2)
    assigned_fold = fold1

    y_array = y.to_numpy()
    fold_loads = [int(((assigned_fold == fold) & eligible).sum()) for fold in range(n_folds)]
    fold_pos = [int((((assigned_fold == fold) & eligible) & (y_array == 1)).sum()) for fold in range(n_folds)]
    fold_neg = [int((((assigned_fold == fold) & eligible) & (y_array == 0)).sum()) for fold in range(n_folds)]
    eligible_rows = int(eligible.sum())
    mixed_rows = int(mixed.sum())
    empty_folds = sum(load == 0 for load in fold_loads)
    degenerate_folds = sum((p == 0 or n == 0) and (p + n > 0) for p, n in zip(fold_pos, fold_neg))
    score = eligible_rows - 2.0 * float(np.std(fold_loads)) - 10.0 * empty_folds - 6.0 * degenerate_folds + 0.5 * min(fold_loads)
    return SplitAssignment(n_folds, dict(amine_to_fold), fold_loads, eligible_rows, mixed_rows, score)


def choose_primary_assignment(df: pd.DataFrame) -> SplitAssignment:
    amines = sorted((set(df["amine1_name_canonical"]) | set(df["amine2_name_canonical"])) - {""})
    y = df["y_phase_sep"].astype(int)
    rng = np.random.default_rng(SEED)
    best: SplitAssignment | None = None
    for n_folds in PRIMARY_FOLD_OPTIONS:
        for _ in range(RANDOM_SEARCH_ITERS):
            candidate = {amine: int(rng.integers(0, n_folds)) for amine in amines}
            scored = assignment_score(df, y, candidate, n_folds)
            if best is None or scored.score > best.score:
                best = scored
    if best is None:
        raise RuntimeError("No valid strict amine-object-holdout assignment found")
    return best


def apply_primary_split(df: pd.DataFrame, assignment: SplitAssignment) -> pd.DataFrame:
    out = df.copy()
    out["amine1_primary_bucket"] = out["amine1_name_canonical"].map(assignment.amine_to_fold)
    out["amine2_primary_bucket"] = out["amine2_name_canonical"].map(assignment.amine_to_fold)
    has_a2 = out["amine2_name_canonical"].fillna("") != ""
    eligible = (~has_a2) | (out["amine1_primary_bucket"] == out["amine2_primary_bucket"])
    out["is_primary_evidence_row"] = eligible.astype(int)
    out["primary_evidence_fold"] = np.where(eligible, out["amine1_primary_bucket"], np.nan)
    out["is_secondary_mixed_dual_row"] = ((has_a2) & (~eligible)).astype(int)
    return out


def make_model(model_name: str) -> Pipeline:
    if model_name == "logreg":
        estimator = LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=SEED)
    elif model_name == "svm_rbf":
        estimator = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)
    else:
        raise ValueError(model_name)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def compute_metric_row(y_true: pd.Series, y_score: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    y_true = pd.Series(y_true).astype(int)
    y_score = pd.Series(y_score).astype(float)
    out = {
        "n_rows": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "degenerate_single_class": int(y_true.nunique() < 2),
        "threshold": threshold,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "f1": np.nan,
        "precision": np.nan,
        "recall": np.nan,
    }
    if len(y_true) == 0:
        return out
    if y_true.nunique() >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["average_precision"] = float(average_precision_score(y_true, y_score))
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out["f1"] = float(f1)
    out["precision"] = float(precision)
    out["recall"] = float(recall)
    return out


def run_primary_protocol(df: pd.DataFrame, trainable_cols: list[str], model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = df.loc[df["is_primary_evidence_row"] == 1].copy()
    oof_rows = []
    fold_rows = []
    for fold in sorted(eligible["primary_evidence_fold"].dropna().astype(int).unique()):
        train_df = eligible.loc[eligible["primary_evidence_fold"] != fold].copy()
        test_df = eligible.loc[eligible["primary_evidence_fold"] == fold].copy()
        heldout_amines = (set(test_df["amine1_name_canonical"]) | set(test_df["amine2_name_canonical"])) - {""}
        train_amines = (set(train_df["amine1_name_canonical"]) | set(train_df["amine2_name_canonical"])) - {""}
        if heldout_amines & train_amines:
            raise RuntimeError(f"Primary split leakage detected in fold {fold}")
        pipeline = make_model(model_name)
        pipeline.fit(train_df[trainable_cols], train_df["y_phase_sep"])
        y_score = pipeline.predict_proba(test_df[trainable_cols])[:, 1]
        preds = pd.DataFrame(
            {
                "row_id": test_df["row_id"].values,
                "feature_set_version": FEATURE_SET_VERSION,
                "model_name": model_name,
                "protocol": "primary_evidence",
                "fold_id": fold,
                "y_true": test_df["y_phase_sep"].values,
                "y_score": y_score,
                "y_pred": (y_score >= 0.5).astype(int),
            }
        )
        oof_rows.append(preds)
        metrics = compute_metric_row(test_df["y_phase_sep"], pd.Series(y_score))
        metrics.update({"feature_set_version": FEATURE_SET_VERSION, "model_name": model_name, "protocol": "primary_evidence", "fold_id": fold})
        fold_rows.append(metrics)
    return pd.concat(oof_rows, ignore_index=True), pd.DataFrame(fold_rows)


def run_secondary_protocol(df: pd.DataFrame, trainable_cols: list[str], model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mixed_df = df.loc[df["is_secondary_mixed_dual_row"] == 1].copy()
    if mixed_df.empty or mixed_df["amine_set_key"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()
    splitter = GroupKFold(n_splits=min(5, mixed_df["amine_set_key"].nunique()))
    oof_rows = []
    fold_rows = []
    for fold_id, (_, test_pos) in enumerate(splitter.split(mixed_df, mixed_df["y_phase_sep"], groups=mixed_df["amine_set_key"])):
        test_df = mixed_df.iloc[test_pos].copy()
        test_groups = set(test_df["amine_set_key"])
        train_df = df.loc[~df["amine_set_key"].isin(test_groups)].copy()
        pipeline = make_model(model_name)
        pipeline.fit(train_df[trainable_cols], train_df["y_phase_sep"])
        y_score = pipeline.predict_proba(test_df[trainable_cols])[:, 1]
        preds = pd.DataFrame(
            {
                "row_id": test_df["row_id"].values,
                "feature_set_version": FEATURE_SET_VERSION,
                "model_name": model_name,
                "protocol": "secondary_diagnostic_pair_holdout",
                "fold_id": fold_id,
                "y_true": test_df["y_phase_sep"].values,
                "y_score": y_score,
                "y_pred": (y_score >= 0.5).astype(int),
            }
        )
        oof_rows.append(preds)
        metrics = compute_metric_row(test_df["y_phase_sep"], pd.Series(y_score))
        metrics.update({"feature_set_version": FEATURE_SET_VERSION, "model_name": model_name, "protocol": "secondary_diagnostic_pair_holdout", "fold_id": fold_id})
        fold_rows.append(metrics)
    return pd.concat(oof_rows, ignore_index=True), pd.DataFrame(fold_rows)


def build_stratified_metrics(oof_df: pd.DataFrame) -> pd.DataFrame:
    subgroup_defs = {
        "full": lambda x: pd.Series(True, index=x.index),
        "with_water": lambda x: x["H2O wt%"] > 0,
        "anhydrous": lambda x: x["H2O wt%"] == 0,
        "organic_solvent": lambda x: x["contains_organic_solvent"] == 1,
        "aqueous_only": lambda x: x["is_aqueous_only"] == 1,
        "dual_amine_aqueous": lambda x: x["system_type"] == "dual_amine_aqueous",
        "dual_amine_plus_water": lambda x: (x["has_amine2"] == 1) & (x["H2O wt%"] > 0),
        "single_amine_solvent": lambda x: x["system_type"] == "single_amine_solvent",
        "dual_amine_solvent": lambda x: x["system_type"] == "dual_amine_solvent",
        "single_amine_aqueous": lambda x: x["system_type"] == "single_amine_aqueous",
    }
    rows = []
    for (model_name, protocol), group_df in oof_df.groupby(["model_name", "protocol"]):
        for subgroup_name, subgroup_fn in subgroup_defs.items():
            subset = group_df.loc[subgroup_fn(group_df)].copy()
            metrics = compute_metric_row(subset["y_true"], subset["y_score"])
            metrics.update({"feature_set_version": FEATURE_SET_VERSION, "model_name": model_name, "protocol": protocol, "subgroup": subgroup_name})
            rows.append(metrics)
    return pd.DataFrame(rows)


def main() -> None:
    modeling_df, trainable_cols = load_modeling_inputs()
    assignment = choose_primary_assignment(modeling_df)
    split_df = apply_primary_split(modeling_df, assignment)

    all_oof = []
    all_fold_metrics = []
    for model_name, file_name in [
        ("logreg", "baseline_oof_predictions_logreg_legacy_v1.csv"),
        ("svm_rbf", "baseline_oof_predictions_svm_legacy_v1.csv"),
    ]:
        primary_oof, primary_metrics = run_primary_protocol(split_df, trainable_cols, model_name)
        secondary_oof, secondary_metrics = run_secondary_protocol(split_df, trainable_cols, model_name)
        model_oof = pd.concat([primary_oof, secondary_oof], ignore_index=True)
        model_oof = model_oof.merge(
            split_df[
                [
                    "row_id",
                    "amine1_name_canonical",
                    "amine2_name_canonical",
                    "solvent_name_canonical",
                    "system_type",
                    "contains_organic_solvent",
                    "is_aqueous_only",
                    "has_amine2",
                    "H2O wt%",
                    "is_primary_evidence_row",
                    "is_secondary_mixed_dual_row",
                ]
            ],
            on="row_id",
            how="left",
            validate="many_to_one",
        )
        model_oof.to_csv(OUTPUTS_DIR / file_name, index=False)
        all_oof.append(model_oof)
        all_fold_metrics.append(pd.concat([primary_metrics, secondary_metrics], ignore_index=True))

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    fold_metrics.to_csv(OUTPUTS_DIR / "baseline_fold_metrics_legacy_v1.csv", index=False)
    stratified_metrics = build_stratified_metrics(pd.concat(all_oof, ignore_index=True))
    stratified_metrics.to_csv(OUTPUTS_DIR / "baseline_stratified_metrics_legacy_v1.csv", index=False)


if __name__ == "__main__":
    main()
