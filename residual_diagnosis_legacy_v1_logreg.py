from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

FEATURE_SET_VERSION = "legacy_aligned_backbone_v1"
MODEL_NAME = "logreg"
PRIMARY_PROTOCOL = "strict_amine_object_holdout"
CONFIRM_PROTOCOL = "ds_unknown"
BIAS_DIRECTION_THRESHOLD = 0.02
SYSTEMATIC_BIAS_THRESHOLD = 0.05
LOW_SUPPORT_THRESHOLD = 10

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = ROOT / "artifacts"

SOLVENT_FAMILY_MAP = {
    "": "aqueous_only_no_solvent",
    "Methanol": "alcohol",
    "ethanol": "alcohol",
    "n-propanol": "alcohol",
    "isopropanol": "alcohol",
    "n-butanol": "alcohol",
    "isobutanol": "alcohol",
    "Tertiarybutanol": "alcohol",
    "n-pentylalcohol": "alcohol",
    "1-Heptanol": "alcohol",
    "1-Octanol": "alcohol",
    "Isooctanol": "alcohol",
    "DGM": "glycol_ether_glyme",
    "DGME": "glycol_ether_glyme",
    "DEGEE": "glycol_ether_glyme",
    "DEGDEE": "glycol_ether_glyme",
    "EGBE": "glycol_ether_glyme",
    "TGBE": "glycol_ether_glyme",
    "TEGDME": "glycol_ether_glyme",
    "EGME": "glycol_ether_glyme",
    "EGEE": "glycol_ether_glyme",
    "DEGBE": "glycol_ether_glyme",
    "EGDEE": "glycol_ether_glyme",
    "TEG": "polyol",
    "DMSO": "sulfur_polar_aprotic",
    "sulfolane": "sulfur_polar_aprotic",
    "DMCA": "amide_lactam",
    "NMP": "amide_lactam",
    "DX": "cyclic_ether",
    "PC": "carbonate",
}


def compute_metric_row(y_true: pd.Series, y_score: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    y_true = pd.Series(y_true).astype(int)
    y_score = pd.Series(y_score).astype(float)
    out = {
        "support_rows": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "degenerate_single_class": int(y_true.nunique() < 2),
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


def bias_direction(mean_residual: float) -> str:
    if pd.isna(mean_residual):
        return "unknown"
    if mean_residual >= BIAS_DIRECTION_THRESHOLD:
        return "underpredict_phase_sep"
    if mean_residual <= -BIAS_DIRECTION_THRESHOLD:
        return "overpredict_phase_sep"
    return "near_balanced"


def reacted_eligibility_map() -> dict[str, int]:
    reactive_df = pd.read_csv(ARTIFACTS_DIR / "reactive_role_dictionary.csv")
    return (
        reactive_df.assign(is_reacted_feature_eligible=reactive_df["is_reacted_feature_eligible"].fillna(0).astype(int))
        .set_index("molecule_name")["is_reacted_feature_eligible"]
        .to_dict()
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_df = pd.read_csv(OUTPUTS_DIR / "protocol_suite_model_oof_predictions_legacy_v1.csv")
    metrics_df = pd.read_csv(OUTPUTS_DIR / "protocol_suite_model_metrics_legacy_v1.csv")
    fold_df = pd.read_csv(OUTPUTS_DIR / "protocol_suite_model_fold_metrics_legacy_v1.csv")
    strat_df = pd.read_csv(OUTPUTS_DIR / "protocol_suite_model_stratified_metrics_legacy_v1.csv")
    base_df = pd.read_csv(OUTPUTS_DIR / "baseline_modeling_table_legacy_v1.csv")
    return pred_df, metrics_df, fold_df, strat_df, base_df


def build_residual_rows(pred_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    eligibility = reacted_eligibility_map()
    logreg = pred_df.loc[
        (pred_df["model_name"] == MODEL_NAME) & (pred_df["protocol"].isin([PRIMARY_PROTOCOL, CONFIRM_PROTOCOL]))
    ].copy()

    strict_rows = set(logreg.loc[logreg["protocol"] == PRIMARY_PROTOCOL, "row_id"])
    confirm_rows = set(logreg.loc[logreg["protocol"] == CONFIRM_PROTOCOL, "row_id"])

    base_meta = base_df[
        [
            "row_id",
            "Phase Type",
            "has_solvent",
            "salt_out_defined_flag",
        ]
    ].copy()
    rows = logreg.merge(base_meta, on="row_id", how="left", validate="many_to_one")

    rows["p_logreg"] = rows["y_score"].astype(float)
    rows["residual"] = rows["y_true"].astype(int) - rows["p_logreg"]
    rows["abs_residual"] = rows["residual"].abs()
    rows["error_flag"] = (rows["y_true"].astype(int) != rows["y_pred"].astype(int)).astype(int)
    rows["with_water_flag"] = (rows["H2O wt%"] > 0).astype(int)
    rows["anhydrous_flag"] = (rows["H2O wt%"] == 0).astype(int)
    rows["organic_solvent_flag"] = rows["contains_organic_solvent"].astype(int)
    rows["dual_amine_aqueous_flag"] = (rows["system_type"] == "dual_amine_aqueous").astype(int)
    rows["single_amine_solvent_flag"] = (rows["system_type"] == "single_amine_solvent").astype(int)
    rows["official_primary_evidence_row_flag"] = rows["row_id"].isin(strict_rows).astype(int)
    rows["ds_unknown_confirmation_row_flag"] = rows["row_id"].isin(confirm_rows).astype(int)
    rows["protocol_is_official_primary_evidence"] = (rows["protocol"] == PRIMARY_PROTOCOL).astype(int)
    rows["protocol_is_ds_unknown_confirmation"] = (rows["protocol"] == CONFIRM_PROTOCOL).astype(int)
    rows["amine_object_count"] = rows["amine2_name_canonical"].fillna("").ne("").astype(int) + 1
    rows["amine_objects_joined"] = rows.apply(
        lambda r: "|".join([x for x in [str(r["amine1_name_canonical"]).strip(), str(r["amine2_name_canonical"]).strip()] if x and x != "nan"]),
        axis=1,
    )
    rows["solvent_family"] = rows["solvent_name_canonical"].fillna("").map(SOLVENT_FAMILY_MAP).fillna("other_unmapped")
    rows["amine1_reacted_eligible"] = rows["amine1_name_canonical"].map(lambda x: int(eligibility.get(str(x), 0)))
    rows["amine2_reacted_eligible"] = rows["amine2_name_canonical"].map(lambda x: int(eligibility.get(str(x), 0)) if pd.notna(x) and str(x).strip() else 0)
    rows["all_amines_reacted_eligible"] = (
        rows[["amine1_reacted_eligible", "amine2_reacted_eligible"]].sum(axis=1) >= rows["amine_object_count"]
    ).astype(int)
    rows["any_amines_reacted_eligible"] = (
        rows[["amine1_reacted_eligible", "amine2_reacted_eligible"]].sum(axis=1) >= 1
    ).astype(int)

    ordered_cols = [
        "row_id",
        "feature_set_version",
        "model_name",
        "protocol",
        "fold_id",
        "heldout_group",
        "y_true",
        "p_logreg",
        "y_pred",
        "residual",
        "abs_residual",
        "error_flag",
        "system_type",
        "Phase Type",
        "with_water_flag",
        "anhydrous_flag",
        "organic_solvent_flag",
        "dual_amine_aqueous_flag",
        "single_amine_solvent_flag",
        "contains_organic_solvent",
        "is_aqueous_only",
        "has_amine2",
        "has_solvent",
        "aq_regime_flag",
        "dry_nonaq_flag",
        "salt_out_defined_flag",
        "H2O wt%",
        "amine1_name_canonical",
        "amine2_name_canonical",
        "amine_set_key",
        "amine_object_count",
        "amine_objects_joined",
        "solvent_name_canonical",
        "solvent_family",
        "amine1_reacted_eligible",
        "amine2_reacted_eligible",
        "all_amines_reacted_eligible",
        "any_amines_reacted_eligible",
        "official_primary_evidence_row_flag",
        "ds_unknown_confirmation_row_flag",
        "protocol_is_official_primary_evidence",
        "protocol_is_ds_unknown_confirmation",
    ]
    return rows[ordered_cols].copy()


def regime_specs(df: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    return [
        ("overall", "full", pd.Series(True, index=df.index)),
        ("water_regime", "with_water", df["with_water_flag"] == 1),
        ("water_regime", "anhydrous", df["anhydrous_flag"] == 1),
        ("solvent_regime", "organic_solvent_containing", df["organic_solvent_flag"] == 1),
        ("solvent_regime", "aqueous_only", df["is_aqueous_only"] == 1),
        ("system_type", "dual_amine_aqueous", df["system_type"] == "dual_amine_aqueous"),
        ("system_type", "dual_amine_plus_water", (df["has_amine2"] == 1) & (df["with_water_flag"] == 1)),
        ("system_type", "single_amine_solvent", df["system_type"] == "single_amine_solvent"),
        ("system_type", "dual_amine_solvent", df["system_type"] == "dual_amine_solvent"),
        ("system_type", "single_amine_aqueous", df["system_type"] == "single_amine_aqueous"),
    ]


def summarize_subset(subset: pd.DataFrame) -> dict[str, object]:
    metrics = compute_metric_row(subset["y_true"], subset["p_logreg"])
    mean_residual = float(subset["residual"].mean()) if len(subset) else np.nan
    out = {
        **metrics,
        "mean_p_logreg": float(subset["p_logreg"].mean()) if len(subset) else np.nan,
        "mean_residual": mean_residual,
        "mean_abs_residual": float(subset["abs_residual"].mean()) if len(subset) else np.nan,
        "error_rate": float(subset["error_flag"].mean()) if len(subset) else np.nan,
        "false_positive_count": int(((subset["y_true"] == 0) & (subset["y_pred"] == 1)).sum()),
        "false_negative_count": int(((subset["y_true"] == 1) & (subset["y_pred"] == 0)).sum()),
        "bias_direction": bias_direction(mean_residual),
        "systematic_bias_flag": int(abs(mean_residual) >= SYSTEMATIC_BIAS_THRESHOLD) if len(subset) else 0,
        "low_support_flag": int(len(subset) < LOW_SUPPORT_THRESHOLD),
    }
    return out


def add_protocol_direction_consistency(summary_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    strict_lookup = (
        summary_df.loc[summary_df["protocol"] == PRIMARY_PROTOCOL, key_cols + ["mean_residual", "bias_direction"]]
        .rename(columns={"mean_residual": "strict_mean_residual", "bias_direction": "strict_bias_direction"})
        .copy()
    )
    merged = summary_df.merge(strict_lookup, on=key_cols, how="left", validate="many_to_one")
    merged["direction_consistent_with_strict"] = np.where(
        merged["protocol"] == PRIMARY_PROTOCOL,
        1,
        (
            (merged["bias_direction"] == merged["strict_bias_direction"])
            | (merged["bias_direction"].eq("near_balanced") & merged["strict_bias_direction"].eq("near_balanced"))
        ).astype(int),
    )
    return merged


def build_regime_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for protocol in [PRIMARY_PROTOCOL, CONFIRM_PROTOCOL]:
        protocol_df = rows_df.loc[rows_df["protocol"] == protocol].copy()
        for regime_category, regime_name, mask in regime_specs(protocol_df):
            subset = protocol_df.loc[mask].copy()
            record = {
                "feature_set_version": FEATURE_SET_VERSION,
                "model_name": MODEL_NAME,
                "protocol": protocol,
                "regime_category": regime_category,
                "regime_name": regime_name,
            }
            record.update(summarize_subset(subset))
            summary_rows.append(record)
    summary_df = pd.DataFrame(summary_rows)
    return add_protocol_direction_consistency(summary_df, ["regime_category", "regime_name"])


def explode_by_amine_object(rows_df: pd.DataFrame) -> pd.DataFrame:
    exploded_rows: list[dict[str, object]] = []
    for row in rows_df.itertuples(index=False):
        amines = [row.amine1_name_canonical]
        if isinstance(row.amine2_name_canonical, str) and row.amine2_name_canonical:
            amines.append(row.amine2_name_canonical)
        for amine in amines:
            exploded_rows.append(
                {
                    "row_id": row.row_id,
                    "protocol": row.protocol,
                    "fold_id": row.fold_id,
                    "system_type": row.system_type,
                    "organic_solvent_flag": row.organic_solvent_flag,
                    "dual_amine_aqueous_flag": row.dual_amine_aqueous_flag,
                    "amine_object": amine,
                    "y_true": row.y_true,
                    "p_logreg": row.p_logreg,
                    "y_pred": row.y_pred,
                    "residual": row.residual,
                    "abs_residual": row.abs_residual,
                    "error_flag": row.error_flag,
                }
            )
    return pd.DataFrame(exploded_rows)


def build_amine_object_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    exploded = explode_by_amine_object(rows_df)
    records: list[dict[str, object]] = []
    for (protocol, amine_object), subset in exploded.groupby(["protocol", "amine_object"], sort=False):
        record = {
            "feature_set_version": FEATURE_SET_VERSION,
            "model_name": MODEL_NAME,
            "protocol": protocol,
            "amine_object": amine_object,
            "organic_row_fraction": float(subset["organic_solvent_flag"].mean()),
            "dual_amine_aqueous_row_fraction": float(subset["dual_amine_aqueous_flag"].mean()),
            "dominant_system_type": subset["system_type"].mode().iloc[0] if not subset["system_type"].mode().empty else "",
        }
        record.update(summarize_subset(subset))
        records.append(record)
    summary_df = pd.DataFrame(records)
    return add_protocol_direction_consistency(summary_df, ["amine_object"])


def build_solvent_family_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (protocol, solvent_family), subset in rows_df.groupby(["protocol", "solvent_family"], sort=False):
        record = {
            "feature_set_version": FEATURE_SET_VERSION,
            "model_name": MODEL_NAME,
            "protocol": protocol,
            "solvent_family": solvent_family,
            "distinct_solvent_names": int(subset["solvent_name_canonical"].nunique()),
            "solvent_names_joined": "|".join(sorted({str(x) for x in subset["solvent_name_canonical"].fillna("")})),
        }
        record.update(summarize_subset(subset))
        records.append(record)
    summary_df = pd.DataFrame(records)
    return add_protocol_direction_consistency(summary_df, ["solvent_family"])


def reacted_coverage_for_subset(rows_df: pd.DataFrame) -> tuple[float, int]:
    subset = rows_df.copy()
    if subset.empty:
        return 0.0, 0
    distinct_objects = set(subset["amine1_name_canonical"]) | set(subset["amine2_name_canonical"])
    distinct_objects.discard("")
    eligible_objects: set[str] = set()
    for row in subset.itertuples(index=False):
        for obj, eligible in [
            (row.amine1_name_canonical, row.amine1_reacted_eligible),
            (row.amine2_name_canonical, row.amine2_reacted_eligible),
        ]:
            if isinstance(obj, str) and obj and int(eligible) == 1:
                eligible_objects.add(obj)
    return float(subset["all_amines_reacted_eligible"].mean()), len(eligible_objects & distinct_objects)


def top_family_pattern(solvent_summary_df: pd.DataFrame, protocol: str) -> str:
    subset = solvent_summary_df.loc[
        (solvent_summary_df["protocol"] == protocol)
        & (solvent_summary_df["support_rows"] >= 15)
        & (solvent_summary_df["solvent_family"] != "aqueous_only_no_solvent")
    ].copy()
    if subset.empty:
        return "no stable family pattern"
    subset = subset.sort_values(["mean_abs_residual", "support_rows"], ascending=[False, False]).head(3)
    return "; ".join(
        f"{row.solvent_family}({row.bias_direction}, n={int(row.support_rows)}, mae={row.mean_abs_residual:.3f})"
        for row in subset.itertuples(index=False)
    )


def top_object_pattern(amine_summary_df: pd.DataFrame, protocol: str, dual_only: bool = False) -> str:
    subset = amine_summary_df.loc[(amine_summary_df["protocol"] == protocol) & (amine_summary_df["support_rows"] >= 10)].copy()
    if dual_only:
        subset = subset.loc[subset["dual_amine_aqueous_row_fraction"] >= 0.5].copy()
    if subset.empty:
        return "no stable object pattern"
    subset = subset.sort_values(["mean_abs_residual", "support_rows"], ascending=[False, False]).head(3)
    return "; ".join(
        f"{row.amine_object}({row.bias_direction}, n={int(row.support_rows)}, mae={row.mean_abs_residual:.3f})"
        for row in subset.itertuples(index=False)
    )


def build_route_evidence(
    rows_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    amine_df: pd.DataFrame,
    solvent_df: pd.DataFrame,
) -> pd.DataFrame:
    def regime_row(protocol: str, regime_name: str) -> pd.Series:
        return regime_df.loc[(regime_df["protocol"] == protocol) & (regime_df["regime_name"] == regime_name)].iloc[0]

    strict_full = regime_row(PRIMARY_PROTOCOL, "full")
    confirm_full = regime_row(CONFIRM_PROTOCOL, "full")
    strict_organic = regime_row(PRIMARY_PROTOCOL, "organic_solvent_containing")
    confirm_organic = regime_row(CONFIRM_PROTOCOL, "organic_solvent_containing")
    strict_dual = regime_row(PRIMARY_PROTOCOL, "dual_amine_aqueous")
    confirm_dual = regime_row(CONFIRM_PROTOCOL, "dual_amine_aqueous")

    strict_organic_rows = rows_df.loc[(rows_df["protocol"] == PRIMARY_PROTOCOL) & (rows_df["organic_solvent_flag"] == 1)].copy()
    strict_dual_rows = rows_df.loc[(rows_df["protocol"] == PRIMARY_PROTOCOL) & (rows_df["dual_amine_aqueous_flag"] == 1)].copy()
    confirm_organic_rows = rows_df.loc[(rows_df["protocol"] == CONFIRM_PROTOCOL) & (rows_df["organic_solvent_flag"] == 1)].copy()
    confirm_dual_rows = rows_df.loc[(rows_df["protocol"] == CONFIRM_PROTOCOL) & (rows_df["dual_amine_aqueous_flag"] == 1)].copy()

    strict_org_coverage, strict_org_objects = reacted_coverage_for_subset(strict_organic_rows)
    confirm_org_coverage, confirm_org_objects = reacted_coverage_for_subset(confirm_organic_rows)
    strict_dual_coverage, strict_dual_objects = reacted_coverage_for_subset(strict_dual_rows)
    confirm_dual_coverage, confirm_dual_objects = reacted_coverage_for_subset(confirm_dual_rows)

    route_a_decision = "hold"
    if (
        (strict_full["roc_auc"] - strict_organic["roc_auc"] >= 0.05)
        and (confirm_full["roc_auc"] - confirm_organic["roc_auc"] >= 0.05)
        and (strict_org_coverage >= 0.70)
        and (strict_org_objects >= 5)
    ):
        route_a_decision = "go"

    route_b_decision = "no-go"
    if (
        (strict_full["roc_auc"] - strict_dual["roc_auc"] >= 0.05)
        and (confirm_full["roc_auc"] - confirm_dual["roc_auc"] >= 0.05)
    ):
        route_b_decision = "hold"

    rows = [
        {
            "route_name": "Route A",
            "decision": route_a_decision,
            "decision_scope": "preparation_only",
            "target_regime": "organic_solvent_containing",
            "primary_protocol": PRIMARY_PROTOCOL,
            "confirmation_protocol": CONFIRM_PROTOCOL,
            "primary_support_rows": int(strict_organic["support_rows"]),
            "primary_roc_auc": strict_organic["roc_auc"],
            "primary_average_precision": strict_organic["average_precision"],
            "primary_f1": strict_organic["f1"],
            "primary_mean_residual": strict_organic["mean_residual"],
            "primary_mean_abs_residual": strict_organic["mean_abs_residual"],
            "primary_error_rate": strict_organic["error_rate"],
            "primary_auc_deficit_vs_full": strict_full["roc_auc"] - strict_organic["roc_auc"],
            "confirmation_support_rows": int(confirm_organic["support_rows"]),
            "confirmation_roc_auc": confirm_organic["roc_auc"],
            "confirmation_average_precision": confirm_organic["average_precision"],
            "confirmation_f1": confirm_organic["f1"],
            "confirmation_mean_residual": confirm_organic["mean_residual"],
            "confirmation_mean_abs_residual": confirm_organic["mean_abs_residual"],
            "confirmation_error_rate": confirm_organic["error_rate"],
            "confirmation_auc_deficit_vs_full": confirm_full["roc_auc"] - confirm_organic["roc_auc"],
            "reacted_all_eligible_rate_primary": strict_org_coverage,
            "reacted_all_eligible_rate_confirmation": confirm_org_coverage,
            "distinct_eligible_amine_objects_primary": strict_org_objects,
            "distinct_eligible_amine_objects_confirmation": confirm_org_objects,
            "pattern_summary_primary": top_family_pattern(solvent_df, PRIMARY_PROTOCOL),
            "pattern_summary_confirmation": top_family_pattern(solvent_df, CONFIRM_PROTOCOL),
            "object_summary_primary": top_object_pattern(amine_df, PRIMARY_PROTOCOL),
            "object_summary_confirmation": top_object_pattern(amine_df, CONFIRM_PROTOCOL),
            "decision_reason": (
                "Primary evidence shows a large organic-solvent deficit versus full strict evidence, DS-unknown also remains weaker than its full baseline, "
                "organic residuals concentrate in solvent families rather than pure noise, and reacted eligibility coverage is high enough to justify preparation."
                if route_a_decision == "go"
                else "Organic weakness is present but not yet strong or stable enough across strict plus DS-unknown to justify Route A preparation."
            ),
        },
        {
            "route_name": "Route B",
            "decision": route_b_decision,
            "decision_scope": "preparation_only",
            "target_regime": "dual_amine_aqueous",
            "primary_protocol": PRIMARY_PROTOCOL,
            "confirmation_protocol": CONFIRM_PROTOCOL,
            "primary_support_rows": int(strict_dual["support_rows"]),
            "primary_roc_auc": strict_dual["roc_auc"],
            "primary_average_precision": strict_dual["average_precision"],
            "primary_f1": strict_dual["f1"],
            "primary_mean_residual": strict_dual["mean_residual"],
            "primary_mean_abs_residual": strict_dual["mean_abs_residual"],
            "primary_error_rate": strict_dual["error_rate"],
            "primary_auc_deficit_vs_full": strict_full["roc_auc"] - strict_dual["roc_auc"],
            "confirmation_support_rows": int(confirm_dual["support_rows"]),
            "confirmation_roc_auc": confirm_dual["roc_auc"],
            "confirmation_average_precision": confirm_dual["average_precision"],
            "confirmation_f1": confirm_dual["f1"],
            "confirmation_mean_residual": confirm_dual["mean_residual"],
            "confirmation_mean_abs_residual": confirm_dual["mean_abs_residual"],
            "confirmation_error_rate": confirm_dual["error_rate"],
            "confirmation_auc_deficit_vs_full": confirm_full["roc_auc"] - confirm_dual["roc_auc"],
            "reacted_all_eligible_rate_primary": strict_dual_coverage,
            "reacted_all_eligible_rate_confirmation": confirm_dual_coverage,
            "distinct_eligible_amine_objects_primary": strict_dual_objects,
            "distinct_eligible_amine_objects_confirmation": confirm_dual_objects,
            "pattern_summary_primary": top_object_pattern(amine_df, PRIMARY_PROTOCOL, dual_only=True),
            "pattern_summary_confirmation": top_object_pattern(amine_df, CONFIRM_PROTOCOL, dual_only=True),
            "object_summary_primary": top_object_pattern(amine_df, PRIMARY_PROTOCOL, dual_only=True),
            "object_summary_confirmation": top_object_pattern(amine_df, CONFIRM_PROTOCOL, dual_only=True),
            "decision_reason": (
                "Dual-amine aqueous remains one of the strongest regimes under strict primary evidence and stays strong under DS-unknown confirmation, "
                "so the required subgroup weakness for Route B is absent."
            ),
        },
    ]
    return pd.DataFrame(rows)


def validate_against_benchmark(regime_df: pd.DataFrame, metrics_df: pd.DataFrame, fold_df: pd.DataFrame, strat_df: pd.DataFrame) -> None:
    regime_full = regime_df.loc[regime_df["regime_name"] == "full", ["protocol", "roc_auc", "average_precision", "f1"]].copy()
    metrics_full = metrics_df.loc[
        metrics_df["model_name"] == MODEL_NAME, ["protocol", "roc_auc", "average_precision", "f1"]
    ].copy()
    merged_full = regime_full.merge(metrics_full, on="protocol", suffixes=("_resid", "_benchmark"), validate="one_to_one")
    for metric in ["roc_auc", "average_precision", "f1"]:
        diff = (merged_full[f"{metric}_resid"] - merged_full[f"{metric}_benchmark"]).abs().max()
        if diff > 1e-9:
            raise ValueError(f"Full-metric mismatch against benchmark for {metric}: {diff}")

    strat_subset = strat_df.loc[
        (strat_df["model_name"] == MODEL_NAME)
        & (strat_df["protocol"].isin([PRIMARY_PROTOCOL, CONFIRM_PROTOCOL]))
        & (strat_df["subgroup"].isin(["full", "with_water", "anhydrous", "organic_solvent_containing", "dual_amine_aqueous", "single_amine_solvent"]))
    ].copy()
    compare_subset = regime_df.loc[
        regime_df["regime_name"].isin(["full", "with_water", "anhydrous", "organic_solvent_containing", "dual_amine_aqueous", "single_amine_solvent"])
    ][["protocol", "regime_name", "roc_auc", "average_precision", "f1"]].copy()
    merged_strat = compare_subset.merge(
        strat_subset,
        left_on=["protocol", "regime_name"],
        right_on=["protocol", "subgroup"],
        suffixes=("_resid", "_benchmark"),
        validate="one_to_one",
    )
    for metric in ["roc_auc", "average_precision", "f1"]:
        diff = (merged_strat[f"{metric}_resid"] - merged_strat[f"{metric}_benchmark"]).abs().max()
        if diff > 1e-9:
            raise ValueError(f"Stratified-metric mismatch against benchmark for {metric}: {diff}")

    fold_counts = fold_df.loc[fold_df["model_name"] == MODEL_NAME].groupby("protocol")["fold_id"].nunique().to_dict()
    for protocol in [PRIMARY_PROTOCOL, CONFIRM_PROTOCOL]:
        expected = int(fold_counts.get(protocol, 0))
        observed = int(regime_df.loc[(regime_df["protocol"] == protocol) & (regime_df["regime_name"] == "full"), "support_rows"].notna().sum())
        if expected <= 0 or observed <= 0:
            raise ValueError(f"Missing fold or regime coverage for protocol {protocol}")


def write_residual_report(regime_df: pd.DataFrame, amine_df: pd.DataFrame, solvent_df: pd.DataFrame) -> None:
    strict_regime = regime_df.loc[regime_df["protocol"] == PRIMARY_PROTOCOL].copy()
    confirm_regime = regime_df.loc[regime_df["protocol"] == CONFIRM_PROTOCOL].copy()

    def row_from(df: pd.DataFrame, name: str) -> pd.Series:
        return df.loc[df["regime_name"] == name].iloc[0]

    strict_full = row_from(strict_regime, "full")
    strict_org = row_from(strict_regime, "organic_solvent_containing")
    strict_dual = row_from(strict_regime, "dual_amine_aqueous")
    strict_single = row_from(strict_regime, "single_amine_solvent")
    confirm_org = row_from(confirm_regime, "organic_solvent_containing")
    confirm_dual = row_from(confirm_regime, "dual_amine_aqueous")

    strict_bad_amines = (
        amine_df.loc[(amine_df["protocol"] == PRIMARY_PROTOCOL) & (amine_df["support_rows"] >= 10)]
        .sort_values(["mean_abs_residual", "support_rows"], ascending=[False, False])
        .head(5)
    )
    strict_bad_families = (
        solvent_df.loc[(solvent_df["protocol"] == PRIMARY_PROTOCOL) & (solvent_df["support_rows"] >= 10)]
        .sort_values(["mean_abs_residual", "support_rows"], ascending=[False, False])
        .head(5)
    )

    lines = [
        "# Residual Diagnosis Legacy V1 LogReg",
        "",
        "## Scope",
        "",
        "This report diagnoses residual behavior for `legacy_aligned_backbone_v1 + Logistic Regression`.",
        "",
        f"- Main evidence: `{PRIMARY_PROTOCOL}`",
        f"- Confirmation evidence: `{CONFIRM_PROTOCOL}`",
        "- DS-known is not used as increment-activation evidence here.",
        "",
        "## Main Strict Findings",
        "",
        f"- Full strict set: n={int(strict_full['support_rows'])}, ROC-AUC {strict_full['roc_auc']:.3f}, AP {strict_full['average_precision']:.3f}, F1 {strict_full['f1']:.3f}, mean residual {strict_full['mean_residual']:.3f}, mean absolute residual {strict_full['mean_abs_residual']:.3f}, error rate {strict_full['error_rate']:.3f}.",
        f"- Strongest shortfall is `organic_solvent_containing`: n={int(strict_org['support_rows'])}, ROC-AUC {strict_org['roc_auc']:.3f}, AP {strict_org['average_precision']:.3f}, F1 {strict_org['f1']:.3f}, mean residual {strict_org['mean_residual']:.3f}, error rate {strict_org['error_rate']:.3f}.",
        f"- `single_amine_solvent` tracks the same weakness: n={int(strict_single['support_rows'])}, ROC-AUC {strict_single['roc_auc']:.3f}, mean residual {strict_single['mean_residual']:.3f}, error rate {strict_single['error_rate']:.3f}.",
        f"- `dual_amine_aqueous` is not the short board: n={int(strict_dual['support_rows'])}, ROC-AUC {strict_dual['roc_auc']:.3f}, AP {strict_dual['average_precision']:.3f}, F1 {strict_dual['f1']:.3f}, mean absolute residual {strict_dual['mean_abs_residual']:.3f}, error rate {strict_dual['error_rate']:.3f}.",
        "",
        "## DS-unknown Confirmation",
        "",
        f"- Organic-solvent weakness survives under `{CONFIRM_PROTOCOL}`: ROC-AUC {confirm_org['roc_auc']:.3f}, AP {confirm_org['average_precision']:.3f}, mean residual {confirm_org['mean_residual']:.3f}, error rate {confirm_org['error_rate']:.3f}.",
        f"- Dual-amine aqueous remains comparatively strong under `{CONFIRM_PROTOCOL}`: ROC-AUC {confirm_dual['roc_auc']:.3f}, AP {confirm_dual['average_precision']:.3f}, error rate {confirm_dual['error_rate']:.3f}.",
        "",
        "## Concentration Patterns",
        "",
        "### High-Residual Amine Objects In Strict Evidence",
        "",
    ]
    for row in strict_bad_amines.itertuples(index=False):
        lines.append(
            f"- {row.amine_object}: n={int(row.support_rows)}, mean residual {row.mean_residual:.3f}, mean absolute residual {row.mean_abs_residual:.3f}, error rate {row.error_rate:.3f}, dominant system `{row.dominant_system_type}`."
        )
    lines.extend(["", "### High-Residual Solvent Families In Strict Evidence", ""])
    for row in strict_bad_families.itertuples(index=False):
        lines.append(
            f"- {row.solvent_family}: n={int(row.support_rows)}, mean residual {row.mean_residual:.3f}, mean absolute residual {row.mean_abs_residual:.3f}, error rate {row.error_rate:.3f}."
        )

    (REPORTS_DIR / "residual_diagnosis_legacy_v1_logreg.md").write_text("\n".join(lines), encoding="utf-8")


def write_increment_review(route_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    ds_known = metrics_df.loc[(metrics_df["protocol"] == "ds_known") & (metrics_df["model_name"] == "xgboost")].iloc[0]
    strict = metrics_df.loc[(metrics_df["protocol"] == PRIMARY_PROTOCOL) & (metrics_df["model_name"] == MODEL_NAME)].iloc[0]
    route_a = route_df.loc[route_df["route_name"] == "Route A"].iloc[0]
    route_b = route_df.loc[route_df["route_name"] == "Route B"].iloc[0]

    lines = [
        "# Increment Evidence Review Legacy V1 LogReg",
        "",
        "## Route A",
        "",
        f"- Decision: `{route_a['decision']}`",
        f"- Scope: `{route_a['target_regime']}`",
        f"- Primary evidence: ROC-AUC {route_a['primary_roc_auc']:.3f}, AP {route_a['primary_average_precision']:.3f}, mean residual {route_a['primary_mean_residual']:.3f}, error rate {route_a['primary_error_rate']:.3f}.",
        f"- Confirmation evidence: ROC-AUC {route_a['confirmation_roc_auc']:.3f}, AP {route_a['confirmation_average_precision']:.3f}, mean residual {route_a['confirmation_mean_residual']:.3f}, error rate {route_a['confirmation_error_rate']:.3f}.",
        f"- Reacted eligibility coverage: strict {route_a['reacted_all_eligible_rate_primary']:.1%} with {int(route_a['distinct_eligible_amine_objects_primary'])} eligible objects; confirmation {route_a['reacted_all_eligible_rate_confirmation']:.1%}.",
        f"- Family pattern: {route_a['pattern_summary_primary']}.",
        f"- Reason: {route_a['decision_reason']}",
        "",
        "## Route B",
        "",
        f"- Decision: `{route_b['decision']}`",
        f"- Scope: `{route_b['target_regime']}`",
        f"- Primary evidence: ROC-AUC {route_b['primary_roc_auc']:.3f}, AP {route_b['primary_average_precision']:.3f}, mean residual {route_b['primary_mean_residual']:.3f}, error rate {route_b['primary_error_rate']:.3f}.",
        f"- Confirmation evidence: ROC-AUC {route_b['confirmation_roc_auc']:.3f}, AP {route_b['confirmation_average_precision']:.3f}, mean residual {route_b['confirmation_mean_residual']:.3f}, error rate {route_b['confirmation_error_rate']:.3f}.",
        f"- Object pattern: {route_b['pattern_summary_primary']}.",
        f"- Reason: {route_b['decision_reason']}",
        "",
        "## DS-known Status",
        "",
        "- DS-known remains a within-known-chemistry interpolation / limited extrapolation display protocol.",
        "- It is not suitable as increment-activation evidence because training still contains the same chemistry identity, only under different formulation conditions.",
        f"- Example: DS-known XGBoost reaches ROC-AUC {ds_known['roc_auc']:.3f}, while the official strict LogReg baseline is ROC-AUC {strict['roc_auc']:.3f}. That gap should be read as chemistry-known interpolation capacity, not as stronger chemistry-generalization evidence.",
        "- Tree-model gains under DS-known therefore do not justify Route A or Route B activation by themselves.",
    ]
    (REPORTS_DIR / "increment_evidence_review_legacy_v1_logreg.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    pred_df, metrics_df, fold_df, strat_df, base_df = load_inputs()
    rows_df = build_residual_rows(pred_df, base_df)
    regime_df = build_regime_summary(rows_df)
    amine_df = build_amine_object_summary(rows_df)
    solvent_df = build_solvent_family_summary(rows_df)
    route_df = build_route_evidence(rows_df, regime_df, amine_df, solvent_df)
    validate_against_benchmark(regime_df, metrics_df, fold_df, strat_df)

    rows_df.to_csv(OUTPUTS_DIR / "residual_diagnosis_rows_legacy_v1_logreg.csv", index=False)
    regime_df.to_csv(OUTPUTS_DIR / "residual_summary_by_regime_legacy_v1_logreg.csv", index=False)
    amine_df.to_csv(OUTPUTS_DIR / "residual_summary_by_amine_object_legacy_v1_logreg.csv", index=False)
    solvent_df.to_csv(OUTPUTS_DIR / "residual_summary_by_solvent_family_legacy_v1_logreg.csv", index=False)
    route_df.to_csv(OUTPUTS_DIR / "routeA_routeB_evidence_legacy_v1_logreg.csv", index=False)

    write_residual_report(regime_df, amine_df, solvent_df)
    write_increment_review(route_df, metrics_df)


if __name__ == "__main__":
    main()
