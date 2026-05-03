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

SEED = 20260417
PRIMARY_FOLD_OPTIONS = [5, 4, 3]
RANDOM_SEARCH_ITERS = 800

ROOT = Path(__file__).resolve().parents[1]
DATA_INTERMEDIATE_DIR = ROOT / "data_intermediate"
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = ROOT / "artifacts"


@dataclass
class SplitAssignment:
    n_folds: int
    amine_to_fold: dict[str, int]
    fold_loads: list[int]
    eligible_rows: int
    mixed_rows: int
    score: float


def ensure_dirs() -> None:
    for path in [DATA_INTERMEDIATE_DIR, OUTPUTS_DIR]:
        path.mkdir(exist_ok=True)


def build_canonical_map() -> dict[str, str]:
    canon_df = pd.read_csv(ARTIFACTS_DIR / "name_canonicalization_table.csv")
    mapping: dict[str, str] = {}
    for row in canon_df.itertuples(index=False):
        raw_name = str(row.raw_name).strip()
        canonical_name = str(row.canonical_name).strip()
        if raw_name in mapping and mapping[raw_name] != canonical_name:
            raise ValueError(f"Conflicting canonicalization for {raw_name}")
        mapping[raw_name] = canonical_name
    return mapping


def canonicalize_value(value: object, mapping: dict[str, str]) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return mapping.get(text, text)


def parse_base_name(job_name: str) -> str:
    text = str(job_name)
    if "__" not in text:
        return text
    parts = text.split("__")
    if parts[0] == "neutral" and len(parts) >= 2:
        return parts[1]
    if parts[0] == "reacted" and len(parts) >= 3:
        return parts[1]
    return text


def build_main_table(mapping: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "final_modeling_dataset.csv").copy()
    df.insert(0, "row_id", [f"R{i:06d}" for i in range(1, len(df) + 1)])
    df["amine1_name_canonical"] = df["Amine 1"].map(lambda x: canonicalize_value(x, mapping))
    df["amine2_name_canonical"] = df["Amine 2"].map(lambda x: canonicalize_value(x, mapping))
    df["solvent_name_canonical"] = df["Solvent"].map(lambda x: canonicalize_value(x, mapping))
    df["water_name_canonical"] = "H2O"
    df["amine_set_key"] = df.apply(
        lambda row: "|".join(sorted([x for x in [row["amine1_name_canonical"], row["amine2_name_canonical"]] if x])),
        axis=1,
    )
    df["amine_object_count"] = df["amine_set_key"].map(lambda x: 0 if not x else len(x.split("|")))
    return df


def compress_sigma_profiles(mapping: dict[str, str]) -> pd.DataFrame:
    wide = pd.read_csv(ROOT / "sigma_profiles_wide_selected.csv")
    wide = wide.loc[wide["source_type"] == "neutral"].copy()
    wide["molecule_name_canonical"] = wide["job_name"].map(parse_base_name).map(lambda x: canonicalize_value(x, mapping))

    sigma_cols = [f"sigma_{i}" for i in range(1, 61)]
    area_cols = [f"area_{i}" for i in range(1, 61)]
    sigma = wide[sigma_cols].astype(float).to_numpy()
    area = wide[area_cols].astype(float).to_numpy()
    area_sum = area.sum(axis=1)

    neg_mask = sigma < -0.008
    nonpolar_mask = np.abs(sigma) <= 0.008
    pos_mask = sigma > 0.008

    sigma_neg = np.where(area_sum > 0, (area * neg_mask).sum(axis=1) / area_sum, np.nan)
    sigma_nonpolar = np.where(area_sum > 0, (area * nonpolar_mask).sum(axis=1) / area_sum, np.nan)
    sigma_pos = np.where(area_sum > 0, (area * pos_mask).sum(axis=1) / area_sum, np.nan)
    sigma_abs_mean = np.where(area_sum > 0, (area * np.abs(sigma)).sum(axis=1) / area_sum, np.nan)
    sigma_sq_mean = np.where(area_sum > 0, (area * np.square(sigma)).sum(axis=1) / area_sum, np.nan)

    return pd.DataFrame(
        {
            "molecule_name_canonical": wide["molecule_name_canonical"],
            "sigma_neg_frac": sigma_neg,
            "sigma_nonpolar_frac": sigma_nonpolar,
            "sigma_pos_frac": sigma_pos,
            "sigma_balance": sigma_pos - sigma_neg,
            "sigma_abs_mean": sigma_abs_mean,
            "sigma_sq_mean": sigma_sq_mean,
        }
    )


def build_neutral_descriptor_table(mapping: dict[str, str]) -> pd.DataFrame:
    surface = pd.read_csv(ROOT / "surface_descriptors_selected.csv")
    surface = surface.loc[surface["source_type"] == "neutral"].copy()
    surface["molecule_name_canonical"] = surface["job_name"].map(parse_base_name).map(lambda x: canonicalize_value(x, mapping))

    surface_cols = [
        "molecule_name_canonical",
        "job_name",
        "dGsolv_kcal_mol",
        "energy_dielectric",
        "area_total",
        "volume",
        "sigma_moment_2",
        "sigma_moment_3",
        "sigma_moment_4",
        "hb_donor_moment_3",
        "hb_acceptor_moment_3",
    ]
    descriptors = surface[surface_cols].copy()
    descriptors["hb_balance"] = descriptors["hb_donor_moment_3"] - descriptors["hb_acceptor_moment_3"]
    descriptors = descriptors.merge(compress_sigma_profiles(mapping), on="molecule_name_canonical", how="left", validate="one_to_one")
    return descriptors


def role_aware_join(main_df: pd.DataFrame, neutral_df: pd.DataFrame) -> pd.DataFrame:
    joined = main_df.copy()
    join_slots = [
        ("amine1_name_canonical", "amine1"),
        ("amine2_name_canonical", "amine2"),
        ("solvent_name_canonical", "solvent"),
        ("water_name_canonical", "water"),
    ]
    for key_col, prefix in join_slots:
        slot_df = neutral_df.add_prefix(f"{prefix}_")
        joined = joined.merge(
            slot_df,
            left_on=key_col,
            right_on=f"{prefix}_molecule_name_canonical",
            how="left",
            validate="many_to_one",
        )
        joined[f"{prefix}_neutral_join_found"] = joined[f"{prefix}_job_name"].notna().astype(int)
    return joined


def safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    return num / denom.replace(0, np.nan)


def weighted_mean(df: pd.DataFrame, slots: list[str], value_col: str, weight_alias: str = "wt_frac") -> pd.Series:
    weights = np.column_stack([df[f"{slot}_{weight_alias}"].fillna(0.0).to_numpy(dtype=float) for slot in slots])
    values = np.column_stack([df[f"{slot}_{value_col}"].to_numpy(dtype=float) for slot in slots])
    weighted = np.nan_to_num(values) * weights
    valid_weight = (~np.isnan(values)).astype(float) * weights
    denom = valid_weight.sum(axis=1)
    num = weighted.sum(axis=1)
    out = np.full(len(df), np.nan, dtype=float)
    np.divide(num, denom, out=out, where=denom > 0)
    return pd.Series(out, index=df.index)


def dual_slot_gap(df: pd.DataFrame, left_col: str, right_col: str) -> pd.Series:
    left = df[left_col].astype(float)
    right = df[right_col].astype(float)
    return (left - right).abs().where(left.notna() & right.notna(), np.nan)


def build_feature_table(joined_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[tuple[str, str]]]:
    df = joined_df.copy()

    df["amine1_wt_frac"] = df["Amine 1 wt%"] / 100.0
    df["amine2_wt_frac"] = df["Amine 2 wt%"] / 100.0
    df["solvent_wt_frac"] = df["Solvent wt%"] / 100.0
    df["water_wt_frac"] = df["H2O wt%"] / 100.0

    feature_rows: list[dict[str, object]] = []
    trainable_features: list[str] = []

    def add_feature(name: str, values: pd.Series, layer: str, source_file: str, source_columns: str, derivation: str) -> None:
        df[name] = values
        feature_rows.append(
            {
                "column_name": name,
                "feature_block": "baseline_backbone",
                "feature_layer": layer,
                "trainable": 1,
                "source_file": source_file,
                "source_columns": source_columns,
                "derivation": derivation,
                "allowed_use": "baseline_only",
            }
        )
        trainable_features.append(name)

    add_feature("l0_amine1_wt_frac", df["amine1_wt_frac"], "L0", "final_modeling_dataset.csv", "Amine 1 wt%", "Amine 1 weight fraction")
    add_feature("l0_amine2_wt_frac", df["amine2_wt_frac"], "L0", "final_modeling_dataset.csv", "Amine 2 wt%", "Amine 2 weight fraction")
    add_feature("l0_solvent_wt_frac", df["solvent_wt_frac"], "L0", "final_modeling_dataset.csv", "Solvent wt%", "Solvent weight fraction")
    add_feature("l0_water_wt_frac", df["water_wt_frac"], "L0", "final_modeling_dataset.csv", "H2O wt%", "Water weight fraction")
    total_amine_frac = df["amine1_wt_frac"] + df["amine2_wt_frac"]
    add_feature("l0_total_amine_wt_frac", total_amine_frac, "L0", "final_modeling_dataset.csv", "Amine 1 wt%; Amine 2 wt%", "Total amine weight fraction")
    add_feature("l0_amine_frac_max", pd.concat([df["amine1_wt_frac"], df["amine2_wt_frac"]], axis=1).max(axis=1), "L0", "final_modeling_dataset.csv", "Amine 1 wt%; Amine 2 wt%", "Maximum amine fraction")
    add_feature("l0_amine_frac_min", pd.concat([df["amine1_wt_frac"], df["amine2_wt_frac"]], axis=1).min(axis=1), "L0", "final_modeling_dataset.csv", "Amine 1 wt%; Amine 2 wt%", "Minimum amine fraction")
    add_feature("l0_amine_frac_gap", (df["amine1_wt_frac"] - df["amine2_wt_frac"]).abs(), "L0", "final_modeling_dataset.csv", "Amine 1 wt%; Amine 2 wt%", "Absolute amine fraction gap")

    frac_cols = ["amine1_wt_frac", "amine2_wt_frac", "solvent_wt_frac", "water_wt_frac"]
    frac_values = df[frac_cols].fillna(0.0).to_numpy(dtype=float)
    entropy_terms = np.zeros_like(frac_values)
    positive_mask = frac_values > 0
    entropy_terms[positive_mask] = frac_values[positive_mask] * np.log(frac_values[positive_mask])
    entropy = -np.sum(entropy_terms, axis=1)
    add_feature("l1_temperature_K", df["Temperature(K)"], "L1", "final_modeling_dataset.csv", "Temperature(K)", "Raw temperature in Kelvin")
    add_feature("l1_inv_temperature", 1.0 / df["Temperature(K)"], "L1", "final_modeling_dataset.csv", "Temperature(K)", "Inverse temperature")
    add_feature("l1_composition_entropy_4part", pd.Series(entropy, index=df.index), "L1", "final_modeling_dataset.csv", "wt fractions", "Four-part composition entropy")
    add_feature("l1_temperature_x_entropy", df["Temperature(K)"] * entropy, "L1", "final_modeling_dataset.csv", "Temperature(K); wt fractions", "Temperature times four-part entropy")
    add_feature("l1_total_amine_frac_x_temperature", total_amine_frac * df["Temperature(K)"], "L1", "final_modeling_dataset.csv", "Amine wt fractions; Temperature(K)", "Total amine fraction times temperature")
    add_feature("l1_water_frac_x_temperature", df["water_wt_frac"] * df["Temperature(K)"], "L1", "final_modeling_dataset.csv", "H2O wt%; Temperature(K)", "Water fraction times temperature")
    add_feature("l1_solvent_frac_x_temperature", df["solvent_wt_frac"] * df["Temperature(K)"], "L1", "final_modeling_dataset.csv", "Solvent wt%; Temperature(K)", "Solvent fraction times temperature")

    weighted_amine_mw = safe_divide(df["Amine1_MW"] * df["amine1_wt_frac"] + df["Amine2_MW"].fillna(0.0) * df["amine2_wt_frac"], total_amine_frac)
    add_feature("l2_amine1_MW", df["Amine1_MW"], "L2", "final_modeling_dataset.csv", "Amine1_MW", "Mother-table amine 1 molecular weight")
    add_feature("l2_amine2_MW", df["Amine2_MW"], "L2", "final_modeling_dataset.csv", "Amine2_MW", "Mother-table amine 2 molecular weight")
    add_feature("l2_solvent_MW", df["Solv_MW"], "L2", "final_modeling_dataset.csv", "Solv_MW", "Mother-table solvent molecular weight")
    add_feature("l2_weighted_amine_MW", weighted_amine_mw, "L2", "final_modeling_dataset.csv", "Amine MWs; wt fractions", "Amine-fraction-weighted molecular weight")
    add_feature("l2_mw_ratio_amine1_to_solvent", safe_divide(df["Amine1_MW"], df["Solv_MW"]), "L2", "final_modeling_dataset.csv", "Amine1_MW; Solv_MW", "Amine 1 MW over solvent MW")
    add_feature("l2_mw_ratio_total_amine_to_solvent", safe_divide(weighted_amine_mw, df["Solv_MW"]), "L2", "final_modeling_dataset.csv", "Amine MWs; Solv_MW; wt fractions", "Weighted amine MW over solvent MW")
    add_feature("l2_mw_ratio_total_amine_to_water", safe_divide(weighted_amine_mw, pd.Series(18.01528, index=df.index)), "L2", "final_modeling_dataset.csv", "Amine MWs; wt fractions", "Weighted amine MW over water MW constant")
    add_feature("l2_wt_ratio_total_amine_to_solvent", safe_divide(total_amine_frac, df["solvent_wt_frac"]), "L2", "final_modeling_dataset.csv", "wt fractions", "Total amine fraction over solvent fraction")
    add_feature("l2_wt_ratio_total_amine_to_water", safe_divide(total_amine_frac, df["water_wt_frac"]), "L2", "final_modeling_dataset.csv", "wt fractions", "Total amine fraction over water fraction")

    l3_specs = [
        ("dGsolv_kcal_mol", "weighted_dGsolv_mean"),
        ("energy_dielectric", "weighted_energy_dielectric_mean"),
        ("area_total", "weighted_area_mean"),
        ("volume", "weighted_volume_mean"),
        ("sigma_moment_2", "weighted_sigma_moment_2_mean"),
        ("sigma_moment_3", "weighted_sigma_moment_3_mean"),
        ("sigma_moment_4", "weighted_sigma_moment_4_mean"),
        ("hb_donor_moment_3", "weighted_hb_donor_moment_3_mean"),
        ("hb_acceptor_moment_3", "weighted_hb_acceptor_moment_3_mean"),
        ("hb_balance", "weighted_hb_balance_mean"),
        ("sigma_neg_frac", "weighted_sigma_neg_frac_mean"),
        ("sigma_nonpolar_frac", "weighted_sigma_nonpolar_frac_mean"),
        ("sigma_pos_frac", "weighted_sigma_pos_frac_mean"),
        ("sigma_balance", "weighted_sigma_balance_mean"),
        ("sigma_abs_mean", "weighted_sigma_abs_mean"),
        ("sigma_sq_mean", "weighted_sigma_sq_mean"),
    ]
    for base_col, out_name in l3_specs:
        add_feature(
            f"l3_{out_name}",
            weighted_mean(df, ["amine1", "amine2", "solvent", "water"], base_col),
            "L3",
            "surface_descriptors_selected.csv; sigma_profiles_wide_selected.csv",
            base_col,
            f"Slot-weighted neutral mean of {base_col}",
        )

    def normalized_weighted_mean(value_col: str, group_slots: list[str]) -> pd.Series:
        weights = np.column_stack([df[f"{slot}_wt_frac"].fillna(0.0).to_numpy(dtype=float) for slot in group_slots])
        values = np.column_stack([df[f"{slot}_{value_col}"].to_numpy(dtype=float) for slot in group_slots])
        weighted = np.nan_to_num(values) * weights
        valid_weight = (~np.isnan(values)).astype(float) * weights
        denom = valid_weight.sum(axis=1)
        out = np.full(len(df), np.nan, dtype=float)
        np.divide(weighted.sum(axis=1), denom, out=out, where=denom > 0)
        return pd.Series(out, index=df.index)

    amine_dg = normalized_weighted_mean("dGsolv_kcal_mol", ["amine1", "amine2"])
    medium_dg = normalized_weighted_mean("dGsolv_kcal_mol", ["solvent", "water"])
    amine_hb = normalized_weighted_mean("hb_balance", ["amine1", "amine2"])
    medium_hb = normalized_weighted_mean("hb_balance", ["solvent", "water"])
    amine_sigma = normalized_weighted_mean("sigma_balance", ["amine1", "amine2"])
    medium_sigma = normalized_weighted_mean("sigma_balance", ["solvent", "water"])

    add_feature("l3_amine_vs_medium_dGsolv_gap", amine_dg - medium_dg, "L3", "surface_descriptors_selected.csv", "dGsolv_kcal_mol", "Amine weighted dGsolv minus medium weighted dGsolv")
    add_feature("l3_amine_vs_medium_hb_balance_gap", amine_hb - medium_hb, "L3", "surface_descriptors_selected.csv", "hb_balance", "Amine weighted H-bond balance minus medium weighted H-bond balance")
    add_feature("l3_amine_vs_medium_sigma_balance_gap", amine_sigma - medium_sigma, "L3", "sigma_profiles_wide_selected.csv", "sigma_balance", "Amine weighted sigma balance minus medium weighted sigma balance")
    add_feature("l3_amine_pair_surface_gap", dual_slot_gap(df, "amine1_dGsolv_kcal_mol", "amine2_dGsolv_kcal_mol"), "L3", "surface_descriptors_selected.csv", "amine1_dGsolv_kcal_mol; amine2_dGsolv_kcal_mol", "Absolute dGsolv gap between amine slots")
    add_feature("l3_amine_pair_sigma_balance_gap", dual_slot_gap(df, "amine1_sigma_balance", "amine2_sigma_balance"), "L3", "sigma_profiles_wide_selected.csv", "amine1_sigma_balance; amine2_sigma_balance", "Absolute sigma-balance gap between amine slots")

    removed_features: list[tuple[str, str]] = []
    final_trainable: list[str] = []
    for col in trainable_features:
        non_null = df[col].dropna()
        if non_null.empty:
            removed_features.append((col, "all_missing"))
            continue
        if non_null.nunique() <= 1:
            removed_features.append((col, "constant"))
            continue
        final_trainable.append(col)

    feature_df = pd.DataFrame(feature_rows)
    feature_df["missing_count"] = feature_df["column_name"].map(df.isna().sum().to_dict())
    feature_df["is_constant"] = feature_df["column_name"].map(lambda x: int(x in {n for n, r in removed_features if r == "constant"}))
    feature_df["is_all_missing"] = feature_df["column_name"].map(lambda x: int(x in {n for n, r in removed_features if r == "all_missing"}))
    feature_df["retained_as_trainable"] = feature_df["column_name"].isin(final_trainable).astype(int)

    meta_columns = [
        "row_id",
        "y_phase_sep",
        "Phase separation",
        "Phase Type",
        "phase_type_B",
        "phase_type_M",
        "phase_type_I",
        "Amine 1",
        "Amine 2",
        "Solvent",
        "Amine 1 SMILES",
        "Amine 2 SMILES",
        "Solvent SMILES",
        "amine1_name_canonical",
        "amine2_name_canonical",
        "solvent_name_canonical",
        "water_name_canonical",
        "CO2_Loading(mol/mol amine)",
        "co2_full_absorption_flag",
        "contains_organic_solvent",
        "is_aqueous_only",
        "system_type",
        "has_amine2",
        "has_solvent",
        "amine_count",
        "nonzero_component_count",
        "amine_set_key",
        "amine_object_count",
        "H2O wt%",
    ]
    meta_dict = pd.DataFrame(
        [
            {
                "column_name": col,
                "feature_block": "meta_only",
                "feature_layer": "meta",
                "trainable": 0,
                "source_file": "final_modeling_dataset.csv",
                "source_columns": col,
                "derivation": "Retained metadata column",
                "allowed_use": "audit_or_split_only",
                "missing_count": int(df[col].isna().sum()),
                "is_constant": int(df[col].dropna().nunique() <= 1),
                "is_all_missing": int(df[col].dropna().empty),
                "retained_as_trainable": 0,
            }
            for col in meta_columns
        ]
    )
    feature_df = pd.concat([feature_df, meta_dict], ignore_index=True)

    baseline_cols = meta_columns + final_trainable
    return df[baseline_cols].copy(), feature_df.sort_values(["feature_block", "feature_layer", "column_name"]), final_trainable, removed_features


def row_fold(row: pd.Series, amine_to_fold: dict[str, int]) -> tuple[bool, int | None]:
    amines = [x for x in [row["amine1_name_canonical"], row["amine2_name_canonical"]] if x]
    if not amines:
        return False, None
    folds = {amine_to_fold[a] for a in amines}
    if len(folds) == 1:
        return True, next(iter(folds))
    return False, None


def assignment_score(df: pd.DataFrame, y: pd.Series, amine_to_fold: dict[str, int], n_folds: int) -> SplitAssignment:
    amine1 = df["amine1_name_canonical"].fillna("").to_numpy()
    amine2 = df["amine2_name_canonical"].fillna("").to_numpy()
    fold1 = np.array([amine_to_fold.get(a, -1) for a in amine1], dtype=int)
    fold2 = np.array([amine_to_fold.get(a, -1) for a in amine2], dtype=int)
    has_a2 = amine2 != ""
    eligible = (~has_a2) | (fold1 == fold2)
    mixed = has_a2 & (fold1 != fold2)
    assigned_fold = fold1

    fold_loads = [int(((assigned_fold == fold) & eligible).sum()) for fold in range(n_folds)]
    fold_pos = [int((((assigned_fold == fold) & eligible) & (y.to_numpy() == 1)).sum()) for fold in range(n_folds)]
    fold_neg = [int((((assigned_fold == fold) & eligible) & (y.to_numpy() == 0)).sum()) for fold in range(n_folds)]
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
        raise RuntimeError("No valid split assignment found")
    return best


def apply_primary_split(df: pd.DataFrame, assignment: SplitAssignment) -> pd.DataFrame:
    out = df.copy()
    out["amine1_primary_bucket"] = out["amine1_name_canonical"].map(assignment.amine_to_fold)
    out["amine2_primary_bucket"] = out["amine2_name_canonical"].map(assignment.amine_to_fold)
    has_a2 = out["amine2_name_canonical"].fillna("") != ""
    eligible = (~has_a2) | (out["amine1_primary_bucket"] == out["amine2_primary_bucket"])
    out["is_primary_evidence_row"] = eligible.astype(int)
    out["primary_evidence_fold"] = np.where(eligible, out["amine1_primary_bucket"], np.nan)
    out["is_secondary_mixed_dual_row"] = ((out["amine_object_count"] >= 2) & (out["is_primary_evidence_row"] == 0)).astype(int)
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
        fold_pred = pd.DataFrame(
            {
                "row_id": test_df["row_id"].values,
                "model_name": model_name,
                "protocol": "primary_evidence",
                "fold_id": fold,
                "y_true": test_df["y_phase_sep"].values,
                "y_score": y_score,
                "y_pred": (y_score >= 0.5).astype(int),
            }
        )
        oof_rows.append(fold_pred)
        metrics = compute_metric_row(test_df["y_phase_sep"], pd.Series(y_score))
        metrics.update({"model_name": model_name, "protocol": "primary_evidence", "fold_id": fold})
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
        fold_pred = pd.DataFrame(
            {
                "row_id": test_df["row_id"].values,
                "model_name": model_name,
                "protocol": "secondary_diagnostic_pair_holdout",
                "fold_id": fold_id,
                "y_true": test_df["y_phase_sep"].values,
                "y_score": y_score,
                "y_pred": (y_score >= 0.5).astype(int),
            }
        )
        oof_rows.append(fold_pred)
        metrics = compute_metric_row(test_df["y_phase_sep"], pd.Series(y_score))
        metrics.update({"model_name": model_name, "protocol": "secondary_diagnostic_pair_holdout", "fold_id": fold_id})
        fold_rows.append(metrics)
    return pd.concat(oof_rows, ignore_index=True), pd.DataFrame(fold_rows)


def build_stratified_metrics(oof_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    if oof_df.empty:
        return pd.DataFrame()
    merged = oof_df.copy()
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
    for (model_name, protocol), group_df in merged.groupby(["model_name", "protocol"]):
        for subgroup_name, subgroup_fn in subgroup_defs.items():
            subset = group_df.loc[subgroup_fn(group_df)].copy()
            metrics = compute_metric_row(subset["y_true"], subset["y_score"])
            metrics.update({"model_name": model_name, "protocol": protocol, "subgroup": subgroup_name})
            rows.append(metrics)
    return pd.DataFrame(rows)


def write_reports(
    baseline_df: pd.DataFrame,
    trainable_cols: list[str],
    removed_features: list[tuple[str, str]],
    feature_df: pd.DataFrame,
    assignment: SplitAssignment,
) -> None:
    layer_counts = feature_df.loc[feature_df["retained_as_trainable"] == 1, "feature_layer"].value_counts().to_dict()
    primary_rows = int((baseline_df["is_primary_evidence_row"] == 1).sum())
    mixed_rows = int((baseline_df["is_secondary_mixed_dual_row"] == 1).sum())
    excluded_lines = "\n".join([f"- `{name}`: {reason}" for name, reason in removed_features]) or "- none"
    trainable_lines = "\n".join([f"- `{col}`" for col in trainable_cols])

    implementation_notes = f"""# Baseline Implementation Notes

## Scope

This step builds the neutral-side baseline table and the evaluation scaffold only. It does not activate Route A or Route B.

## Produced Files

- `data_intermediate/canonicalized_main_table.csv`
- `data_intermediate/neutral_joined_table.csv`
- `outputs/baseline_modeling_table.csv`
- `outputs/baseline_feature_dictionary.csv`
- `outputs/baseline_oof_predictions_logreg.csv`
- `outputs/baseline_oof_predictions_svm.csv`
- `outputs/baseline_fold_metrics.csv`
- `outputs/baseline_stratified_metrics.csv`

## Retained Trainable Feature Count

- total retained trainable columns: `{len(trainable_cols)}`
- layer counts: `{layer_counts}`

## Trainable Columns

{trainable_lines}

## Automatically Removed Candidate Features

{excluded_lines}

## Explicit Rule-Based Exclusions

- reacted absolute descriptor blocks
- raw `sigma_1..60`
- raw `area_1..60`
- reacted routing metadata
- object-level score features
- raw identity name / SMILES shortcuts
- constant-coded `CO2_Loading(mol/mol amine)`
- constant `co2_full_absorption_flag`
- all-missing `dipole_moment`

## Split Summary

- chosen primary fold count: `{assignment.n_folds}`
- primary evidence rows: `{primary_rows}`
- mixed dual-amine rows moved out of primary evidence: `{mixed_rows}`
- fold loads: `{assignment.fold_loads}`
"""
    (REPORTS_DIR / "baseline_implementation_notes.md").write_text(implementation_notes, encoding="utf-8")

    bucket_lines = "\n".join([f"- `{amine}` -> fold `{fold}`" for amine, fold in sorted(assignment.amine_to_fold.items(), key=lambda x: (x[1], x[0]))])
    split_protocol = f"""# Baseline Split Protocol

## Primary Evidence Protocol

Primary evidence uses strict amine-object-holdout. A row enters the primary OOF only when all canonical amine objects in that row map to the same primary bucket.

## Concrete Settings

- chosen fold count: `{assignment.n_folds}`
- random search seed: `{SEED}`
- search iterations per fold-count candidate: `{RANDOM_SEARCH_ITERS}`
- fold loads: `{assignment.fold_loads}`

## Current Bucket Assignment

{bucket_lines}

## Mixed Dual-Amine Handling

- rows with two canonical amines assigned to different primary buckets are removed from the primary OOF
- these rows are tagged `is_secondary_mixed_dual_row = 1`
- they are evaluated only under `secondary_diagnostic_pair_holdout`
- secondary diagnostic outputs are weaker evidence than primary strict amine-object-holdout

## Fold-Safe Preprocessing

- imputation is fit on training folds only
- scaling is fit on training folds only
- no full-data preprocessing is applied
- no target-like aggregation is used
"""
    (REPORTS_DIR / "baseline_split_protocol.md").write_text(split_protocol, encoding="utf-8")


def append_split_metadata_dictionary(feature_df: pd.DataFrame) -> pd.DataFrame:
    split_meta = pd.DataFrame(
        [
            {
                "column_name": "amine1_primary_bucket",
                "feature_block": "meta_only",
                "feature_layer": "meta",
                "trainable": 0,
                "source_file": "derived_split_protocol",
                "source_columns": "amine1_name_canonical",
                "derivation": "Primary split bucket for canonical amine 1",
                "allowed_use": "split_only",
                "missing_count": 0,
                "is_constant": 0,
                "is_all_missing": 0,
                "retained_as_trainable": 0,
            },
            {
                "column_name": "amine2_primary_bucket",
                "feature_block": "meta_only",
                "feature_layer": "meta",
                "trainable": 0,
                "source_file": "derived_split_protocol",
                "source_columns": "amine2_name_canonical",
                "derivation": "Primary split bucket for canonical amine 2",
                "allowed_use": "split_only",
                "missing_count": 0,
                "is_constant": 0,
                "is_all_missing": 0,
                "retained_as_trainable": 0,
            },
            {
                "column_name": "is_primary_evidence_row",
                "feature_block": "meta_only",
                "feature_layer": "meta",
                "trainable": 0,
                "source_file": "derived_split_protocol",
                "source_columns": "amine1_name_canonical; amine2_name_canonical",
                "derivation": "Flag for strict amine-object-holdout eligibility",
                "allowed_use": "split_only",
                "missing_count": 0,
                "is_constant": 0,
                "is_all_missing": 0,
                "retained_as_trainable": 0,
            },
            {
                "column_name": "primary_evidence_fold",
                "feature_block": "meta_only",
                "feature_layer": "meta",
                "trainable": 0,
                "source_file": "derived_split_protocol",
                "source_columns": "amine1_primary_bucket; amine2_primary_bucket",
                "derivation": "Assigned primary fold for strict evidence rows",
                "allowed_use": "split_only",
                "missing_count": 0,
                "is_constant": 0,
                "is_all_missing": 0,
                "retained_as_trainable": 0,
            },
            {
                "column_name": "is_secondary_mixed_dual_row",
                "feature_block": "meta_only",
                "feature_layer": "meta",
                "trainable": 0,
                "source_file": "derived_split_protocol",
                "source_columns": "amine1_primary_bucket; amine2_primary_bucket",
                "derivation": "Flag for mixed dual-amine rows excluded from primary evidence",
                "allowed_use": "split_only",
                "missing_count": 0,
                "is_constant": 0,
                "is_all_missing": 0,
                "retained_as_trainable": 0,
            },
        ]
    )
    return pd.concat([feature_df, split_meta], ignore_index=True)


def main() -> None:
    ensure_dirs()
    mapping = build_canonical_map()
    main_df = build_main_table(mapping)
    neutral_df = build_neutral_descriptor_table(mapping)
    joined_df = role_aware_join(main_df, neutral_df)
    baseline_df, feature_df, trainable_cols, removed_features = build_feature_table(joined_df)
    assignment = choose_primary_assignment(baseline_df)
    baseline_df = apply_primary_split(baseline_df, assignment)
    feature_df = append_split_metadata_dictionary(feature_df)
    baseline_missing = baseline_df.isna().sum().to_dict()
    feature_df.loc[feature_df["column_name"].isin(baseline_missing.keys()), "missing_count"] = feature_df["column_name"].map(baseline_missing)
    feature_df = feature_df.sort_values(["feature_block", "feature_layer", "column_name"]).reset_index(drop=True)

    main_df.to_csv(DATA_INTERMEDIATE_DIR / "canonicalized_main_table.csv", index=False)
    joined_df.to_csv(DATA_INTERMEDIATE_DIR / "neutral_joined_table.csv", index=False)
    baseline_df.to_csv(OUTPUTS_DIR / "baseline_modeling_table.csv", index=False)
    feature_df.to_csv(OUTPUTS_DIR / "baseline_feature_dictionary.csv", index=False)

    all_oof = []
    all_fold_metrics = []
    for model_name, file_name in [("logreg", "baseline_oof_predictions_logreg.csv"), ("svm_rbf", "baseline_oof_predictions_svm.csv")]:
        primary_oof, primary_metrics = run_primary_protocol(baseline_df, trainable_cols, model_name)
        secondary_oof, secondary_metrics = run_secondary_protocol(baseline_df, trainable_cols, model_name)
        model_oof = pd.concat([primary_oof, secondary_oof], ignore_index=True)
        model_oof = model_oof.merge(
            baseline_df[
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
            validate="one_to_one",
        )
        model_oof.to_csv(OUTPUTS_DIR / file_name, index=False)
        all_oof.append(model_oof)
        all_fold_metrics.append(pd.concat([primary_metrics, secondary_metrics], ignore_index=True))

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    fold_metrics.to_csv(OUTPUTS_DIR / "baseline_fold_metrics.csv", index=False)
    stratified_metrics = build_stratified_metrics(pd.concat(all_oof, ignore_index=True), baseline_df)
    stratified_metrics.to_csv(OUTPUTS_DIR / "baseline_stratified_metrics.csv", index=False)

    write_reports(baseline_df, trainable_cols, removed_features, feature_df, assignment)


if __name__ == "__main__":
    main()
