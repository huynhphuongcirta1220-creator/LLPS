from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

FEATURE_SET_VERSION = "legacy_aligned_backbone_v1"
SEED = 20260417
DS_KNOWN_FOLDS = 5
DS_UNKNOWN_FOLDS = 5
STRICT_FOLD_OPTIONS = [5, 4, 3]
STRICT_RANDOM_SEARCH_ITERS = 800

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"

MODEL_ORDER = ["logreg", "svm_rbf", "xgboost", "lightgbm"]
METRIC_COLUMNS = ["roc_auc", "average_precision", "f1", "precision", "recall"]

MODEL_PARAM_TEXT = {
    "logreg": "LogisticRegression(max_iter=5000, solver='liblinear', class_weight='balanced', random_state=20260417)",
    "svm_rbf": "SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=20260417)",
    "xgboost": (
        "XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, "
        "colsample_bytree=0.8, min_child_weight=1.0, reg_lambda=1.0, objective='binary:logistic', "
        "eval_metric='logloss', tree_method='hist', random_state=20260417, n_jobs=1)"
    ),
    "lightgbm": (
        "LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, num_leaves=15, "
        "subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_lambda=0.0, "
        "objective='binary', random_state=20260417, n_jobs=1, verbosity=-1)"
    ),
}


@dataclass
class FoldSpec:
    protocol: str
    fold_id: int
    heldout_group: str
    train_index: pd.Index
    test_index: pd.Index


@dataclass
class ProtocolBundle:
    protocol: str
    fold_unit: str
    folds: list[FoldSpec]
    eligible_mask: pd.Series
    group_key: pd.Series
    heldout_assignments: pd.Series
    n_total_rows: int
    n_scored_rows: int
    n_excluded_rows: int
    n_groups_total: int
    n_groups_scored: int
    n_folds_planned: int
    n_folds_scored: int
    exclusion_reason: str
    mixed_rows_excluded: int
    note: str


@dataclass
class StrictAssignment:
    n_folds: int
    amine_to_fold: dict[str, int]
    fold_loads: list[int]
    eligible_rows: int
    mixed_rows: int
    score: float


def load_modeling_inputs() -> tuple[pd.DataFrame, list[str]]:
    modeling_df = pd.read_csv(OUTPUTS_DIR / "baseline_modeling_table_legacy_v1.csv").copy()
    for col in ["amine1_name_canonical", "amine2_name_canonical", "solvent_name_canonical", "amine_set_key", "system_type"]:
        modeling_df[col] = modeling_df[col].fillna("").astype(str)
    feature_df = pd.read_csv(OUTPUTS_DIR / "baseline_feature_dictionary_legacy_v1.csv")
    trainable_cols = feature_df.loc[feature_df["retained_as_trainable"] == 1, "column_name"].tolist()
    if len(trainable_cols) != 22:
        raise ValueError(f"legacy_v1 trainable feature count drifted from 22: {len(trainable_cols)}")

    modeling_df["chemistry_system_key"] = modeling_df.apply(
        lambda row: f"{row['amine_set_key']}__{row['solvent_name_canonical']}__aq{int(row['aq_regime_flag'])}",
        axis=1,
    )
    modeling_df["amine_object_count"] = modeling_df["amine2_name_canonical"].ne("").astype(int) + 1
    return modeling_df, trainable_cols


def safe_mean(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce")
    if series.notna().sum() == 0:
        return float("nan")
    return float(series.mean())


def safe_std(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce")
    if series.notna().sum() <= 1:
        return 0.0
    return float(series.std(ddof=1))


def safe_divide(num: float, denom: float) -> float:
    if denom == 0 or pd.isna(denom):
        return 0.0
    return float(num / denom)


def contains_any_heldout(df: pd.DataFrame, heldout_amines: set[str]) -> pd.Series:
    return df["amine1_name_canonical"].isin(heldout_amines) | df["amine2_name_canonical"].isin(heldout_amines)


def make_model(model_name: str) -> Pipeline:
    if model_name == "logreg":
        estimator = LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=SEED)
        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)]
    elif model_name == "svm_rbf":
        estimator = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED)
        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)]
    elif model_name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=1,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
    elif model_name == "lightgbm":
        estimator = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_lambda=0.0,
            objective="binary",
            random_state=SEED,
            n_jobs=1,
            verbosity=-1,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
    else:
        raise ValueError(model_name)
    return Pipeline(steps=steps)


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


def build_ds_known_protocol(df: pd.DataFrame) -> ProtocolBundle:
    system_counts = df["chemistry_system_key"].value_counts()
    eligible_mask = df["chemistry_system_key"].map(system_counts).fillna(0).astype(int) >= 2
    fold_assign = pd.Series(np.nan, index=df.index, dtype=float)
    sort_cols = ["inv_T", "x_H2O", "x_Am1", "x_Am2", "x_Sol", "row_id"]
    eligible_df = df.loc[eligible_mask].sort_values(["chemistry_system_key"] + sort_cols)
    for _, system_df in eligible_df.groupby("chemistry_system_key", sort=True):
        ordered_idx = system_df.sort_values(sort_cols).index.tolist()
        for offset, idx in enumerate(ordered_idx):
            fold_assign.loc[idx] = offset % DS_KNOWN_FOLDS

    folds: list[FoldSpec] = []
    for fold_id in range(DS_KNOWN_FOLDS):
        test_index = df.index[(eligible_mask) & (fold_assign == fold_id)]
        if len(test_index) == 0:
            continue
        train_index = df.index.difference(test_index)
        folds.append(
            FoldSpec(
                protocol="ds_known",
                fold_id=fold_id,
                heldout_group=f"fold_{fold_id}",
                train_index=train_index,
                test_index=test_index,
            )
        )

    return ProtocolBundle(
        protocol="ds_known",
        fold_unit="row fold within chemistry_system_key",
        folds=folds,
        eligible_mask=eligible_mask,
        group_key=df["chemistry_system_key"],
        heldout_assignments=fold_assign,
        n_total_rows=len(df),
        n_scored_rows=int(eligible_mask.sum()),
        n_excluded_rows=int((~eligible_mask).sum()),
        n_groups_total=int(df["chemistry_system_key"].nunique()),
        n_groups_scored=int(df.loc[eligible_mask, "chemistry_system_key"].nunique()),
        n_folds_planned=DS_KNOWN_FOLDS,
        n_folds_scored=len(folds),
        exclusion_reason="Excluded singleton chemistry systems because DS-known requires chemistry already observed in training.",
        mixed_rows_excluded=0,
        note="Rows are assigned within order-invariant chemistry systems; training retains same chemistry identity with different wt% or temperature rows.",
    )


def build_ds_unknown_protocol(df: pd.DataFrame) -> ProtocolBundle:
    amines = np.array(sorted((set(df["amine1_name_canonical"]) | set(df["amine2_name_canonical"])) - {""}))
    splitter = KFold(n_splits=DS_UNKNOWN_FOLDS, shuffle=True, random_state=SEED)
    amine_to_fold: dict[str, int] = {}
    for fold_id, (_, test_pos) in enumerate(splitter.split(amines)):
        for pos in test_pos:
            amine_to_fold[str(amines[pos])] = fold_id

    fold1 = df["amine1_name_canonical"].map(amine_to_fold).astype(int)
    has_a2 = df["amine2_name_canonical"].ne("")
    fold2 = df["amine2_name_canonical"].map(amine_to_fold)
    eligible_mask = (~has_a2) | (fold1 == fold2.astype("Int64"))
    assigned_fold = pd.Series(np.where(eligible_mask, fold1, np.nan), index=df.index, dtype=float)

    folds: list[FoldSpec] = []
    for fold_id in range(DS_UNKNOWN_FOLDS):
        heldout_amines = {amine for amine, assigned in amine_to_fold.items() if assigned == fold_id}
        test_index = df.index[(eligible_mask) & (assigned_fold == fold_id)]
        if len(test_index) == 0:
            continue
        train_mask = ~contains_any_heldout(df, heldout_amines)
        train_index = df.index[train_mask]
        folds.append(
            FoldSpec(
                protocol="ds_unknown",
                fold_id=fold_id,
                heldout_group="|".join(sorted(heldout_amines)),
                train_index=train_index,
                test_index=test_index,
            )
        )

    mixed_rows_excluded = int((has_a2 & ~eligible_mask).sum())
    return ProtocolBundle(
        protocol="ds_unknown",
        fold_unit="canonical amine-object k-fold",
        folds=folds,
        eligible_mask=eligible_mask,
        group_key=pd.Series(df["amine1_name_canonical"].map(amine_to_fold), index=df.index),
        heldout_assignments=assigned_fold,
        n_total_rows=len(df),
        n_scored_rows=int(eligible_mask.sum()),
        n_excluded_rows=int((~eligible_mask).sum()),
        n_groups_total=len(amines),
        n_groups_scored=len(amines),
        n_folds_planned=DS_UNKNOWN_FOLDS,
        n_folds_scored=len(folds),
        exclusion_reason="Excluded dual-amine rows whose two canonical amines fall into different unseen-object folds.",
        mixed_rows_excluded=mixed_rows_excluded,
        note="Training excludes any row containing a held-out amine object; ambiguous cross-fold dual rows are not scored.",
    )


def strict_assignment_score(df: pd.DataFrame, y: pd.Series, amine_to_fold: dict[str, int], n_folds: int) -> StrictAssignment:
    amine1 = df["amine1_name_canonical"].to_numpy()
    amine2 = df["amine2_name_canonical"].to_numpy()
    fold1 = np.array([amine_to_fold.get(a, -1) for a in amine1], dtype=int)
    fold2 = np.array([amine_to_fold.get(a, -1) for a in amine2], dtype=int)
    has_a2 = amine2 != ""
    eligible = (~has_a2) | (fold1 == fold2)
    assigned_fold = fold1

    y_array = y.to_numpy()
    fold_loads = [int(((assigned_fold == fold) & eligible).sum()) for fold in range(n_folds)]
    fold_pos = [int((((assigned_fold == fold) & eligible) & (y_array == 1)).sum()) for fold in range(n_folds)]
    fold_neg = [int((((assigned_fold == fold) & eligible) & (y_array == 0)).sum()) for fold in range(n_folds)]
    eligible_rows = int(eligible.sum())
    mixed_rows = int((has_a2 & (fold1 != fold2)).sum())
    empty_folds = sum(load == 0 for load in fold_loads)
    degenerate_folds = sum((p == 0 or n == 0) and (p + n > 0) for p, n in zip(fold_pos, fold_neg))
    score = eligible_rows - 2.0 * float(np.std(fold_loads)) - 10.0 * empty_folds - 6.0 * degenerate_folds + 0.5 * min(fold_loads)
    return StrictAssignment(n_folds, dict(amine_to_fold), fold_loads, eligible_rows, mixed_rows, score)


def choose_strict_assignment(df: pd.DataFrame) -> StrictAssignment:
    amines = sorted((set(df["amine1_name_canonical"]) | set(df["amine2_name_canonical"])) - {""})
    y = df["y_phase_sep"].astype(int)
    rng = np.random.default_rng(SEED)
    best: StrictAssignment | None = None
    for n_folds in STRICT_FOLD_OPTIONS:
        for _ in range(STRICT_RANDOM_SEARCH_ITERS):
            candidate = {amine: int(rng.integers(0, n_folds)) for amine in amines}
            scored = strict_assignment_score(df, y, candidate, n_folds)
            if best is None or scored.score > best.score:
                best = scored
    if best is None:
        raise RuntimeError("No strict amine-object-holdout assignment found")
    return best


def build_strict_holdout_protocol(df: pd.DataFrame) -> ProtocolBundle:
    assignment = choose_strict_assignment(df)
    fold1 = df["amine1_name_canonical"].map(assignment.amine_to_fold).astype(int)
    has_a2 = df["amine2_name_canonical"].ne("")
    fold2 = df["amine2_name_canonical"].map(assignment.amine_to_fold)
    eligible_mask = (~has_a2) | (fold1 == fold2.astype("Int64"))
    assigned_fold = pd.Series(np.where(eligible_mask, fold1, np.nan), index=df.index, dtype=float)

    folds: list[FoldSpec] = []
    eligible_df = df.loc[eligible_mask].copy()
    for fold_id in range(assignment.n_folds):
        test_index = eligible_df.index[assigned_fold.loc[eligible_df.index] == fold_id]
        if len(test_index) == 0:
            continue
        train_index = eligible_df.index[assigned_fold.loc[eligible_df.index] != fold_id]
        heldout_amines = {amine for amine, assigned in assignment.amine_to_fold.items() if assigned == fold_id}
        folds.append(
            FoldSpec(
                protocol="strict_amine_object_holdout",
                fold_id=fold_id,
                heldout_group="|".join(sorted(heldout_amines)),
                train_index=train_index,
                test_index=test_index,
            )
        )

    mixed_rows_excluded = int((has_a2 & ~eligible_mask).sum())
    return ProtocolBundle(
        protocol="strict_amine_object_holdout",
        fold_unit="optimized amine-object holdout fold",
        folds=folds,
        eligible_mask=eligible_mask,
        group_key=pd.Series(df["amine1_name_canonical"].map(assignment.amine_to_fold), index=df.index),
        heldout_assignments=assigned_fold,
        n_total_rows=len(df),
        n_scored_rows=int(eligible_mask.sum()),
        n_excluded_rows=int((~eligible_mask).sum()),
        n_groups_total=len(assignment.amine_to_fold),
        n_groups_scored=len(assignment.amine_to_fold),
        n_folds_planned=assignment.n_folds,
        n_folds_scored=len(folds),
        exclusion_reason="Excluded dual-amine rows spanning different optimized holdout buckets from the primary evidence table.",
        mixed_rows_excluded=mixed_rows_excluded,
        note=f"Official primary evidence protocol. Random-search assignment chose {assignment.n_folds} folds with clean coverage {assignment.eligible_rows}/{len(df)}.",
    )


def build_loco_protocol(df: pd.DataFrame) -> ProtocolBundle:
    amines = sorted((set(df["amine1_name_canonical"]) | set(df["amine2_name_canonical"])) - {""})
    eligible_mask = df["has_amine2"] == 0
    heldout_assignments = pd.Series("", index=df.index, dtype=object)
    folds: list[FoldSpec] = []
    for fold_id, amine in enumerate(amines):
        test_index = df.index[(df["has_amine2"] == 0) & (df["amine1_name_canonical"] == amine)]
        if len(test_index) == 0:
            continue
        heldout_assignments.loc[test_index] = amine
        train_index = df.index[~contains_any_heldout(df, {amine})]
        folds.append(
            FoldSpec(
                protocol="loco",
                fold_id=fold_id,
                heldout_group=amine,
                train_index=train_index,
                test_index=test_index,
            )
        )

    return ProtocolBundle(
        protocol="loco",
        fold_unit="held-out canonical amine object",
        folds=folds,
        eligible_mask=eligible_mask,
        group_key=df["amine1_name_canonical"],
        heldout_assignments=heldout_assignments,
        n_total_rows=len(df),
        n_scored_rows=int(eligible_mask.sum()),
        n_excluded_rows=int((~eligible_mask).sum()),
        n_groups_total=len(amines),
        n_groups_scored=len(folds),
        n_folds_planned=len(amines),
        n_folds_scored=len(folds),
        exclusion_reason="LOCO primary metric scores only single-object rows; mixed dual-amine rows are excluded from the main LOCO score table.",
        mixed_rows_excluded=int((df["has_amine2"] == 1).sum()),
        note="LOCO is the strictest per-object protocol. Amine objects with no single-row support are not scored in the main LOCO metric.",
    )


def validate_fold(protocol: str, df: pd.DataFrame, fold: FoldSpec) -> None:
    train_df = df.loc[fold.train_index]
    test_df = df.loc[fold.test_index]
    if protocol in {"ds_unknown", "strict_amine_object_holdout", "loco"}:
        heldout_amines = {fold.heldout_group} if protocol == "loco" else set(filter(None, fold.heldout_group.split("|")))
        if contains_any_heldout(train_df, heldout_amines).any():
            raise RuntimeError(f"Leakage detected in protocol {protocol}, fold {fold.fold_id}")
    if len(test_df) == 0:
        raise RuntimeError(f"Empty test fold in protocol {protocol}, fold {fold.fold_id}")


def evaluate_protocol(df: pd.DataFrame, trainable_cols: list[str], bundle: ProtocolBundle, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []

    for fold in bundle.folds:
        validate_fold(bundle.protocol, df, fold)
        train_df = df.loc[fold.train_index].copy()
        test_df = df.loc[fold.test_index].copy()
        pipeline = make_model(model_name)
        pipeline.fit(train_df[trainable_cols], train_df["y_phase_sep"])
        y_score = pipeline.predict_proba(test_df[trainable_cols])[:, 1]

        preds = test_df[
            [
                "row_id",
                "amine1_name_canonical",
                "amine2_name_canonical",
                "solvent_name_canonical",
                "amine_set_key",
                "chemistry_system_key",
                "system_type",
                "contains_organic_solvent",
                "is_aqueous_only",
                "has_amine2",
                "H2O wt%",
                "aq_regime_flag",
                "dry_nonaq_flag",
            ]
        ].copy()
        preds["feature_set_version"] = FEATURE_SET_VERSION
        preds["model_name"] = model_name
        preds["protocol"] = bundle.protocol
        preds["fold_id"] = fold.fold_id
        preds["heldout_group"] = fold.heldout_group
        preds["fold_unit"] = bundle.fold_unit
        preds["y_true"] = test_df["y_phase_sep"].to_numpy()
        preds["y_score"] = y_score
        preds["y_pred"] = (y_score >= 0.5).astype(int)
        oof_rows.append(preds)

        metrics = compute_metric_row(test_df["y_phase_sep"], pd.Series(y_score))
        metrics.update(
            {
                "feature_set_version": FEATURE_SET_VERSION,
                "model_name": model_name,
                "protocol": bundle.protocol,
                "fold_id": fold.fold_id,
                "heldout_group": fold.heldout_group,
                "fold_unit": bundle.fold_unit,
                "coverage_rows_total": bundle.n_total_rows,
                "coverage_rows_scored": bundle.n_scored_rows,
                "coverage_rate": safe_divide(bundle.n_scored_rows, bundle.n_total_rows),
            }
        )
        fold_rows.append(metrics)

    return pd.concat(oof_rows, ignore_index=True), pd.DataFrame(fold_rows)


def subgroup_definitions(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "full": pd.Series(True, index=df.index),
        "with_water": df["H2O wt%"] > 0,
        "anhydrous": df["H2O wt%"] == 0,
        "organic_solvent_containing": df["contains_organic_solvent"] == 1,
        "aqueous_only": df["is_aqueous_only"] == 1,
        "dual_amine_aqueous": df["system_type"] == "dual_amine_aqueous",
        "dual_amine_plus_water": (df["has_amine2"] == 1) & (df["H2O wt%"] > 0),
        "single_amine_solvent": df["system_type"] == "single_amine_solvent",
        "dual_amine_solvent": df["system_type"] == "dual_amine_solvent",
        "single_amine_aqueous": df["system_type"] == "single_amine_aqueous",
    }


def build_stratified_metrics(oof_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (protocol, model_name), group_df in oof_df.groupby(["protocol", "model_name"], sort=False):
        subgroup_masks = subgroup_definitions(group_df)
        for subgroup_name, mask in subgroup_masks.items():
            subset = group_df.loc[mask].copy()
            metrics = compute_metric_row(subset["y_true"], subset["y_score"])
            metrics.update(
                {
                    "feature_set_version": FEATURE_SET_VERSION,
                    "protocol": protocol,
                    "model_name": model_name,
                    "subgroup": subgroup_name,
                }
            )
            rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_protocol_metrics(oof_df: pd.DataFrame, fold_df: pd.DataFrame, protocol_bundles: dict[str, ProtocolBundle]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protocol, bundle in protocol_bundles.items():
        for model_name in MODEL_ORDER:
            subset_oof = oof_df.loc[(oof_df["protocol"] == protocol) & (oof_df["model_name"] == model_name)].copy()
            subset_fold = fold_df.loc[(fold_df["protocol"] == protocol) & (fold_df["model_name"] == model_name)].copy()
            overall = compute_metric_row(subset_oof["y_true"], subset_oof["y_score"])
            row = {
                "feature_set_version": FEATURE_SET_VERSION,
                "protocol": protocol,
                "model_name": model_name,
                "fold_unit": bundle.fold_unit,
                "n_total_rows": bundle.n_total_rows,
                "n_scored_rows": bundle.n_scored_rows,
                "coverage_rate": safe_divide(bundle.n_scored_rows, bundle.n_total_rows),
                "n_excluded_rows": bundle.n_excluded_rows,
                "n_groups_total": bundle.n_groups_total,
                "n_groups_scored": bundle.n_groups_scored,
                "n_folds_planned": bundle.n_folds_planned,
                "n_folds_scored": bundle.n_folds_scored,
                "mixed_rows_excluded": bundle.mixed_rows_excluded,
                "exclusion_reason": bundle.exclusion_reason,
                "note": bundle.note,
                "overall_n_rows": overall["n_rows"],
                "overall_n_pos": overall["n_pos"],
                "overall_n_neg": overall["n_neg"],
                "overall_degenerate_single_class": overall["degenerate_single_class"],
                "roc_auc": overall["roc_auc"],
                "average_precision": overall["average_precision"],
                "f1": overall["f1"],
                "precision": overall["precision"],
                "recall": overall["recall"],
            }
            for metric in METRIC_COLUMNS:
                row[f"{metric}_fold_mean"] = safe_mean(subset_fold[metric])
                row[f"{metric}_fold_std"] = safe_std(subset_fold[metric])
                row[f"{metric}_fold_valid"] = int(pd.to_numeric(subset_fold[metric], errors="coerce").notna().sum())
            rows.append(row)
    return pd.DataFrame(rows)


def protocol_coverage_table(protocol_bundles: dict[str, ProtocolBundle]) -> pd.DataFrame:
    rows = []
    for protocol, bundle in protocol_bundles.items():
        rows.append(
            {
                "protocol": protocol,
                "fold_unit": bundle.fold_unit,
                "n_total_rows": bundle.n_total_rows,
                "n_scored_rows": bundle.n_scored_rows,
                "coverage_rate": safe_divide(bundle.n_scored_rows, bundle.n_total_rows),
                "n_excluded_rows": bundle.n_excluded_rows,
                "n_groups_total": bundle.n_groups_total,
                "n_groups_scored": bundle.n_groups_scored,
                "n_folds_planned": bundle.n_folds_planned,
                "n_folds_scored": bundle.n_folds_scored,
                "mixed_rows_excluded": bundle.mixed_rows_excluded,
                "exclusion_reason": bundle.exclusion_reason,
                "note": bundle.note,
            }
        )
    return pd.DataFrame(rows)


def metric_line(label: str, row: pd.Series) -> str:
    return (
        f"- {label}: ROC-AUC {row['roc_auc']:.3f}, AP {row['average_precision']:.3f}, "
        f"F1 {row['f1']:.3f}, Precision {row['precision']:.3f}, Recall {row['recall']:.3f}"
    )


def write_protocol_design_report(protocol_bundles: dict[str, ProtocolBundle]) -> None:
    coverage_df = protocol_coverage_table(protocol_bundles)
    lines = [
        "# Protocol Suite Design",
        "",
        "## Scope",
        "",
        "This document defines the four evaluation protocols used for the controlled model-family benchmark on `legacy_aligned_backbone_v1`.",
        "",
        "All protocols keep the same 22-dimensional neutral-side backbone. No reacted absolute descriptors, routing metadata, or feature-space changes are introduced here.",
        "",
        "## Protocol Definitions",
        "",
        "### DS-known",
        "",
        "- Scientific question: interpolation / limited extrapolation within chemistry systems already seen in training.",
        "- Group key: `chemistry_system_key = sorted(amine set) + solvent + water-presence`.",
        "- Order pollution control: `amine_set_key` is canonical and order-invariant, so `Amine1` / `Amine2` order cannot create separate groups.",
        "- Fold construction: rows are ordered within each chemistry system by `inv_T`, `x_H2O`, `x_Am1`, `x_Am2`, `x_Sol`, `row_id`, then assigned round-robin across 5 folds.",
        "- Mixed dual-amine rows: kept if they belong to a repeated chemistry system; no extra exclusion is needed because DS-known is chemistry-known by design.",
        "- Exclusion rule: singleton chemistry systems are removed from scoring because they cannot satisfy the known-chemistry premise.",
        "",
        "### DS-unknown",
        "",
        "- Scientific question: k-fold generalization to unseen canonical amine objects.",
        "- Group unit: canonical amine object.",
        "- Fold construction: 5-fold deterministic KFold over the canonical amine-object list.",
        "- Test rule: a row is scored only if all amine objects in that row map to the same unseen-object fold.",
        "- Train rule: training excludes any row containing a held-out amine object for that fold.",
        "- Mixed dual-amine rows: rows whose two amines fall into different unseen-object folds are excluded from the main DS-unknown score table.",
        "",
        "### LOCO",
        "",
        "- Scientific question: strict leave-one-canonical-amine-object-out behavior.",
        "- Fold unit: one held-out canonical amine object.",
        "- Test rule: the main LOCO metric scores only single-object rows for the held-out amine.",
        "- Train rule: training excludes every row containing the held-out amine object.",
        "- Mixed dual-amine rows: not silently merged into the main LOCO metric; they are excluded because attribution is ambiguous under per-object leave-out.",
        "",
        "### strict amine-object-holdout",
        "",
        "- Scientific question: strongest clean primary evidence under current anti-leakage rules.",
        "- Fold unit: optimized amine-object holdout bucket.",
        "- Fold construction: random-search amine-to-bucket assignment over candidate fold counts `[5, 4, 3]`, scored for clean coverage and non-degenerate fold balance.",
        "- Test rule: a row enters primary evidence only if every amine object in that row belongs to the same holdout bucket.",
        "- Train rule: official primary fitting uses only the clean primary-evidence rows from other buckets.",
        "- Mixed dual-amine rows: removed from the primary evidence table when they span different optimized buckets.",
        "",
        "## Why These Protocols Fit The Current Project",
        "",
        "- DS-known answers formulation variation questions without pretending it is chemistry-novelty evidence.",
        "- DS-unknown provides k-fold unseen-object evidence without collapsing into one-object-at-a-time LOCO volatility.",
        "- LOCO provides the strictest per-object stress test, but coverage is intentionally lower.",
        "- strict amine-object-holdout remains the official primary evidence because it prioritizes anti-leakage and clean sample assignment.",
        "",
        "## Mixed Dual-Amine Handling",
        "",
        "- Coverage is reported explicitly for every protocol.",
        "- Mixed rows are never silently forced into a leaky primary bucket.",
        "- DS-known keeps mixed rows only when they are legitimate repeated chemistry systems.",
        "- DS-unknown and strict holdout exclude ambiguous cross-fold dual rows from their main score tables.",
        "- LOCO excludes mixed dual rows from the primary metric by construction.",
        "",
        "## Model Family Defaults",
        "",
        "- Logistic Regression uses the existing balanced linear baseline setup for continuity with prior repository outputs.",
        f"- `logreg`: `{MODEL_PARAM_TEXT['logreg']}`",
        f"- `svm_rbf`: `{MODEL_PARAM_TEXT['svm_rbf']}`",
        f"- `xgboost`: `{MODEL_PARAM_TEXT['xgboost']}`",
        f"- `lightgbm`: `{MODEL_PARAM_TEXT['lightgbm']}`",
        "- No hyperparameter search is used in this stage because the goal is controlled family comparison on a fixed backbone, not a tuning contest.",
        "",
        "## Protocol Coverage",
        "",
    ]
    for row in coverage_df.itertuples(index=False):
        lines.extend(
            [
                f"### {row.protocol}",
                "",
                f"- Coverage: {row.n_scored_rows}/{row.n_total_rows} ({row.coverage_rate:.1%})",
                f"- Groups scored: {row.n_groups_scored}/{row.n_groups_total}",
                f"- Folds scored: {row.n_folds_scored}/{row.n_folds_planned}",
                f"- Mixed rows excluded: {row.mixed_rows_excluded}",
                f"- Exclusion reason: {row.exclusion_reason}",
                f"- Note: {row.note}",
                "",
            ]
        )
    (REPORTS_DIR / "protocol_suite_design.md").write_text("\n".join(lines), encoding="utf-8")


def write_benchmark_report(summary_df: pd.DataFrame, stratified_df: pd.DataFrame) -> None:
    phase1_path = OUTPUTS_DIR / "baseline_fold_metrics.csv"
    phase1_note = "Phase1 LogReg sanity comparator unavailable."
    if phase1_path.exists():
        phase1_df = pd.read_csv(phase1_path)
        phase1_primary = phase1_df.loc[(phase1_df["protocol"] == "primary_evidence") & (phase1_df["model_name"] == "logreg")]
        if not phase1_primary.empty:
            phase1_note = (
                "Phase1 LogReg sanity comparator under existing strict primary evidence: "
                f"mean ROC-AUC {safe_mean(phase1_primary['roc_auc']):.3f}, AP {safe_mean(phase1_primary['average_precision']):.3f}, "
                f"F1 {safe_mean(phase1_primary['f1']):.3f}."
            )

    lines = [
        "# Model Family Benchmark On Legacy V1",
        "",
        "## Scope",
        "",
        "This report compares Logistic Regression, SVM-RBF, XGBoost, and LightGBM on the fixed `legacy_aligned_backbone_v1` backbone across DS-known, DS-unknown, LOCO, and strict amine-object-holdout.",
        "",
        "The feature space is fixed at 22 legacy-aligned neutral-side backbone features. No Route A or Route B increment is active here.",
        "",
        "## Default Parameter Policy",
        "",
        "- Only sensible default settings are used.",
        "- No grid search, random search, or Bayesian optimization is performed.",
        "- Minimal numeric-stability choices are limited to solver / tree-method settings and modest tree depth.",
        "- Class weighting remains only in the linear and SVM baselines, consistent with prior repository evaluation outputs.",
        "",
        "## Summary By Protocol",
        "",
    ]
    best_by_protocol: dict[str, pd.Series] = {}
    for protocol in ["ds_known", "ds_unknown", "loco", "strict_amine_object_holdout"]:
        protocol_df = summary_df.loc[summary_df["protocol"] == protocol].copy()
        protocol_df = protocol_df.sort_values(["roc_auc", "average_precision", "f1"], ascending=[False, False, False], na_position="last")
        if protocol_df.empty:
            continue
        best_by_protocol[protocol] = protocol_df.iloc[0]
        lines.append(f"### {protocol}")
        lines.append("")
        for row in protocol_df.itertuples(index=False):
            lines.append(
                f"- {row.model_name}: ROC-AUC {row.roc_auc:.3f}, AP {row.average_precision:.3f}, F1 {row.f1:.3f}, "
                f"fold ROC-AUC {row.roc_auc_fold_mean:.3f} +/- {row.roc_auc_fold_std:.3f}, coverage {row.n_scored_rows}/{row.n_total_rows}"
            )
        lines.append("")

    lines.extend(
        [
            "## Best Model Interpretation",
            "",
            f"- Best DS-known model: `{best_by_protocol['ds_known']['model_name']}`",
            f"- Best DS-unknown model: `{best_by_protocol['ds_unknown']['model_name']}`",
            f"- Best LOCO model: `{best_by_protocol['loco']['model_name']}`",
            f"- Best strict primary-evidence model: `{best_by_protocol['strict_amine_object_holdout']['model_name']}`",
            "",
            "## Official Recommendation",
            "",
            f"- Recommended official baseline model for current primary evidence: `{best_by_protocol['strict_amine_object_holdout']['model_name']}` on `strict_amine_object_holdout`.",
            "- Residual diagnosis should prioritize the official primary-evidence protocol rather than the weaker convenience protocols.",
            "- DS-known remains useful for formulation-space interpolation questions, but not for Route A / Route B activation decisions.",
            "- LOCO remains a stress-test protocol, not the sole basis for increment decisions because coverage is intentionally narrower.",
            "",
            "## Backbone Sensitivity",
            "",
            "- Model-family sensitivity is evaluated here by holding the 22-dimensional legacy backbone fixed.",
            "- If tree ensembles improve only on weaker protocols but not on strict primary evidence, they should not displace the simpler official model.",
            "",
            "## Phase1 Sanity Comparator",
            "",
            f"- {phase1_note}",
            "",
            "## Increment Readiness Implication",
            "",
            "- Route A and Route B should be judged primarily on strict primary evidence residual patterns, then checked against DS-unknown for robustness.",
            "- No increment route should be activated from DS-known-only gains.",
        ]
    )

    interesting_subgroups = ["full", "with_water", "anhydrous", "organic_solvent_containing", "dual_amine_aqueous", "single_amine_solvent"]
    strict_df = stratified_df.loc[stratified_df["protocol"] == "strict_amine_object_holdout"].copy()
    if not strict_df.empty:
        lines.extend(["", "## Strict Protocol Regime Snapshot", ""])
        for subgroup in interesting_subgroups:
            subset = strict_df.loc[strict_df["subgroup"] == subgroup].sort_values(["roc_auc", "average_precision"], ascending=[False, False], na_position="last")
            if subset.empty:
                continue
            best_row = subset.iloc[0]
            lines.append(metric_line(f"{subgroup} best={best_row['model_name']}", best_row))

    (REPORTS_DIR / "model_family_benchmark_legacy_v1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    modeling_df, trainable_cols = load_modeling_inputs()
    protocol_bundles = {
        "ds_known": build_ds_known_protocol(modeling_df),
        "ds_unknown": build_ds_unknown_protocol(modeling_df),
        "loco": build_loco_protocol(modeling_df),
        "strict_amine_object_holdout": build_strict_holdout_protocol(modeling_df),
    }

    all_oof: list[pd.DataFrame] = []
    all_fold_metrics: list[pd.DataFrame] = []
    for protocol_name in ["ds_known", "ds_unknown", "loco", "strict_amine_object_holdout"]:
        bundle = protocol_bundles[protocol_name]
        for model_name in MODEL_ORDER:
            oof_df, fold_df = evaluate_protocol(modeling_df, trainable_cols, bundle, model_name)
            all_oof.append(oof_df)
            all_fold_metrics.append(fold_df)

    oof_predictions = pd.concat(all_oof, ignore_index=True)
    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    stratified_metrics = build_stratified_metrics(oof_predictions)
    summary_metrics = summarize_protocol_metrics(oof_predictions, fold_metrics, protocol_bundles)

    oof_predictions.to_csv(OUTPUTS_DIR / "protocol_suite_model_oof_predictions_legacy_v1.csv", index=False)
    fold_metrics.to_csv(OUTPUTS_DIR / "protocol_suite_model_fold_metrics_legacy_v1.csv", index=False)
    stratified_metrics.to_csv(OUTPUTS_DIR / "protocol_suite_model_stratified_metrics_legacy_v1.csv", index=False)
    summary_metrics.to_csv(OUTPUTS_DIR / "protocol_suite_model_metrics_legacy_v1.csv", index=False)

    write_protocol_design_report(protocol_bundles)
    write_benchmark_report(summary_metrics, stratified_metrics)


if __name__ == "__main__":
    main()
