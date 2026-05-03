from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SIGMA_ACCEPTOR_THRESHOLD = 0.0084
SIGMA_DONOR_THRESHOLD = -0.0084
PRIMARY_PROTOCOL = "strict_amine_object_holdout"
TARGET_SYSTEM_TYPE = "single_amine_solvent"
TARGET_FAMILIES = ["alcohol", "sulfur_polar_aprotic"]

if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    ROOT = Path.cwd()
    if not (ROOT / "outputs").exists() and (ROOT.parent / "outputs").exists():
        ROOT = ROOT.parent

OUTPUTS_DIR = ROOT / "outputs"
ARTIFACTS_DIR = ROOT / "artifacts"

ROUTEA_FAMILY_MAP = {
    "alcohol": "alcohol",
    "sulfur_polar_aprotic": "sulfur_polar_aprotic",
    "glycol_ether_glyme": "glycol_ether_glyme",
}


def build_canonical_map() -> dict[str, str]:
    canon_df = pd.read_csv(ARTIFACTS_DIR / "name_canonicalization_table.csv")
    mapping: dict[str, str] = {}
    for row in canon_df.itertuples(index=False):
        mapping[str(row.raw_name).strip()] = str(row.canonical_name).strip()
    return mapping


def canonicalize_value(value: object, mapping: dict[str, str]) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return mapping.get(text, text)


def parse_job_name(job_name: str) -> tuple[str, str, str]:
    text = str(job_name).strip()
    if text.startswith("neutral__"):
        _, name = text.split("__", 1)
        return "neutral", name, ""
    if text.startswith("reacted__"):
        _, name, form = text.split("__", 2)
        return "reacted", name, form
    raise ValueError(f"Unrecognized job_name format: {job_name}")


def compute_sigma_segments(df: pd.DataFrame) -> pd.DataFrame:
    sigma_cols = [f"sigma_{i}" for i in range(1, 61)]
    area_cols = [f"area_{i}" for i in range(1, 61)]
    sigma_axis = df[sigma_cols].iloc[0].astype(float).to_numpy()
    area_matrix = df[area_cols].astype(float).to_numpy()
    total_area = area_matrix.sum(axis=1)
    area_fraction = np.divide(area_matrix, total_area[:, None], out=np.zeros_like(area_matrix), where=total_area[:, None] > 0)
    acceptor_mask = sigma_axis >= SIGMA_ACCEPTOR_THRESHOLD
    donor_mask = sigma_axis <= SIGMA_DONOR_THRESHOLD
    nonpolar_mask = (~acceptor_mask) & (~donor_mask)

    out = df[["job_name", "source_type"]].copy()
    out["Am_acc"] = (area_fraction * acceptor_mask).sum(axis=1)
    out["Am_don"] = (area_fraction * donor_mask).sum(axis=1)
    out["Am_np"] = (area_fraction * nonpolar_mask).sum(axis=1)
    out["Am_wid"] = (area_fraction * np.square(sigma_axis)).sum(axis=1)
    return out


def build_descriptor_table(mapping: dict[str, str]) -> pd.DataFrame:
    sigma = pd.read_csv(ROOT / "sigma_profiles_wide_selected.csv").copy()
    seg = compute_sigma_segments(sigma)
    parsed = seg["job_name"].map(parse_job_name).tolist()
    seg["descriptor_state"] = [item[0] for item in parsed]
    seg["molecule_name"] = [canonicalize_value(item[1], mapping) for item in parsed]
    seg["reacted_form"] = [item[2] for item in parsed]
    return seg


def routed_form(channel: str, has_carbamate: int, has_bicarbonate: int) -> str:
    if channel == "carbamate_capable" and has_carbamate:
        return "carbamate_pair"
    if channel == "protonation_base_assisted" and has_bicarbonate:
        return "bicarbonate_pair"
    return ""


def aligned_term_record(neutral_row: pd.Series, reacted_row: pd.Series, prefix: str) -> dict[str, float]:
    record: dict[str, float] = {}
    for base in ["Am_acc", "Am_don", "Am_np", "Am_wid"]:
        record[f"{prefix}_{base}_rea"] = float(reacted_row[base])
        record[f"{prefix}_{base}_neu"] = float(neutral_row[base])
        record[f"{prefix}_Delta_{base}"] = float(reacted_row[base] - neutral_row[base])
        record[f"{prefix}_AbsDelta_{base}"] = float(abs(reacted_row[base] - neutral_row[base]))
        denom = max(abs(float(neutral_row[base])), 1e-12)
        record[f"{prefix}_LogRatio_{base}"] = float(np.log(max(float(reacted_row[base]), 1e-12) / denom))
    return record


def build_alignment_audit(descriptor_df: pd.DataFrame) -> pd.DataFrame:
    role_df = pd.read_csv(ARTIFACTS_DIR / "reactive_role_dictionary.csv").copy()
    neutral_df = descriptor_df.loc[descriptor_df["descriptor_state"] == "neutral"].copy()
    reacted_df = descriptor_df.loc[descriptor_df["descriptor_state"] == "reacted"].copy()
    neutral_lookup = neutral_df.set_index("molecule_name")
    rows: list[dict[str, object]] = []

    for role_row in role_df.itertuples(index=False):
        molecule = str(role_row.molecule_name)
        neutral_row = neutral_lookup.loc[molecule]
        if isinstance(neutral_row, pd.DataFrame):
            neutral_row = neutral_row.iloc[0]
        object_reacted = reacted_df.loc[reacted_df["molecule_name"] == molecule].copy()
        forms_present = sorted(set(object_reacted["reacted_form"]))
        has_carbamate = int("carbamate_pair" in forms_present)
        has_bicarbonate = int("bicarbonate_pair" in forms_present)
        main_form = routed_form(str(role_row.reacted_channel_main), has_carbamate, has_bicarbonate)
        aux_form = routed_form(str(role_row.reacted_channel_aux), has_carbamate, has_bicarbonate)

        record: dict[str, object] = {
            "molecule_name": molecule,
            "source_roles_observed": role_row.source_roles_observed,
            "reactive_role": role_row.reactive_role,
            "reacted_channel_main": role_row.reacted_channel_main,
            "reacted_channel_aux": role_row.reacted_channel_aux,
            "is_reacted_feature_eligible": int(role_row.is_reacted_feature_eligible),
            "reacted_forms_observed": "|".join(forms_present) if forms_present else "none",
            "reacted_form_count": int(len(forms_present)),
            "has_carbamate_pair": has_carbamate,
            "has_bicarbonate_pair": has_bicarbonate,
            "main_aligned_form": main_form or "none",
            "aux_aligned_form": aux_form or "none",
            "main_alignment_ready": int(bool(main_form)),
            "aux_alignment_ready": int(bool(aux_form)),
            "explicit_transition_ready": int(role_row.reactive_role == "reactive_amine" and bool(main_form)),
            "Am_acc_neu": float(neutral_row["Am_acc"]),
            "Am_don_neu": float(neutral_row["Am_don"]),
            "Am_np_neu": float(neutral_row["Am_np"]),
            "Am_wid_neu": float(neutral_row["Am_wid"]),
        }

        for form in ["bicarbonate_pair", "carbamate_pair"]:
            form_row = object_reacted.loc[object_reacted["reacted_form"] == form]
            form_prefix = "bicarbonate" if form == "bicarbonate_pair" else "carbamate"
            record[f"{form_prefix}_form_available"] = int(not form_row.empty)
            if not form_row.empty:
                record.update(aligned_term_record(neutral_row, form_row.iloc[0], form_prefix))
            else:
                for base in ["Am_acc", "Am_don", "Am_np", "Am_wid"]:
                    record[f"{form_prefix}_{base}_rea"] = np.nan
                    record[f"{form_prefix}_{base}_neu"] = float(neutral_row[base])
                    record[f"{form_prefix}_Delta_{base}"] = np.nan
                    record[f"{form_prefix}_AbsDelta_{base}"] = np.nan
                    record[f"{form_prefix}_LogRatio_{base}"] = np.nan

        for branch_name, form in [("main", main_form), ("aux", aux_form)]:
            if not form:
                for base in ["Am_acc", "Am_don", "Am_np", "Am_wid"]:
                    record[f"{branch_name}_{base}_rea"] = np.nan
                    record[f"{branch_name}_{base}_neu"] = float(neutral_row[base])
                    record[f"{branch_name}_Delta_{base}"] = np.nan
            else:
                prefix = "bicarbonate" if form == "bicarbonate_pair" else "carbamate"
                for base in ["Am_acc", "Am_don", "Am_np", "Am_wid"]:
                    record[f"{branch_name}_{base}_rea"] = record[f"{prefix}_{base}_rea"]
                    record[f"{branch_name}_{base}_neu"] = record[f"{prefix}_{base}_neu"]
                    record[f"{branch_name}_Delta_{base}"] = record[f"{prefix}_Delta_{base}"]

        rows.append(record)

    return pd.DataFrame(rows)


def scoped_route_rows() -> pd.DataFrame:
    route_df = pd.read_csv(OUTPUTS_DIR / "routeA_family_aware_increment_ready_table.csv").copy()
    route_df = route_df.loc[route_df["protocol"] == PRIMARY_PROTOCOL].copy()
    route_df["routeA2_family"] = route_df["solvent_family"].map(ROUTEA_FAMILY_MAP).fillna("other_organic_family")
    route_df = route_df.loc[
        (route_df["system_type"] == TARGET_SYSTEM_TYPE)
        & (route_df["routeA2_family"].isin(TARGET_FAMILIES))
    ].copy()
    return route_df


def build_feature_table(alignment_df: pd.DataFrame) -> pd.DataFrame:
    route_df = scoped_route_rows()
    lookup = alignment_df.set_index("molecule_name").to_dict(orient="index")
    rows: list[dict[str, object]] = []

    for row in route_df.itertuples(index=False):
        amines = [str(row.amine1_name_canonical).strip()]
        if isinstance(row.amine2_name_canonical, str) and row.amine2_name_canonical.strip():
            amines.append(row.amine2_name_canonical.strip())
        records = [lookup.get(name, {}) for name in amines]
        rows.append(
            {
                "feature_set_version": "routeA2_explicit_transition_phase1",
                "protocol": row.protocol,
                "fold_id": row.fold_id,
                "routeA2_family": row.routeA2_family,
                "system_type": row.system_type,
                "row_id": row.row_id,
                "amine1_name_canonical": row.amine1_name_canonical,
                "amine2_name_canonical": row.amine2_name_canonical,
                "amine_objects_joined": "|".join(amines),
                "n_amine_objects": len(amines),
                "y_true": row.y_true,
                "p_logreg_baseline": row.p_logreg,
                "baseline_residual": row.residual,
                "solvent_name_canonical": row.solvent_name_canonical,
                "solvent_family": row.solvent_family,
                "main_aligned_object_name": amines[0] if amines else "",
                "main_aligned_form": str(records[0].get("main_aligned_form", "none")) if records else "none",
                "main_reactive_role": str(records[0].get("reactive_role", "")) if records else "",
                "main_reacted_channel_main": str(records[0].get("reacted_channel_main", "")) if records else "",
                "main_alignment_ready": int(records[0].get("main_alignment_ready", 0)) if records else 0,
                "explicit_transition_ready": int(records[0].get("explicit_transition_ready", 0)) if records else 0,
                "row_any_aux_alignment_ready": int(any(int(rec.get("aux_alignment_ready", 0)) == 1 for rec in records)),
                "row_any_nonreactive_bypass": int(any(str(rec.get("reactive_role", "")) == "nonreactive_solvent_like" for rec in records)),
                "row_any_uncovered_reactive": int(any(str(rec.get("reactive_role", "")) == "reactive_amine" and int(rec.get("main_alignment_ready", 0)) == 0 for rec in records)),
                "old_routeA_ra_routed_main_delta_mean_avg": row.ra_routed_main_delta_mean_avg,
                "old_routeA_ra_routed_main_delta_maxabs_avg": row.ra_routed_main_delta_maxabs_avg,
                "old_routeA_ra_routed_main_delta_gap_avg": row.ra_routed_main_delta_gap_avg,
                "main_Am_acc_neu": records[0].get("main_Am_acc_neu", np.nan) if records else np.nan,
                "main_Am_don_neu": records[0].get("main_Am_don_neu", np.nan) if records else np.nan,
                "main_Am_np_neu": records[0].get("main_Am_np_neu", np.nan) if records else np.nan,
                "main_Am_wid_neu": records[0].get("main_Am_wid_neu", np.nan) if records else np.nan,
                "main_Am_acc_rea": records[0].get("main_Am_acc_rea", np.nan) if records else np.nan,
                "main_Am_don_rea": records[0].get("main_Am_don_rea", np.nan) if records else np.nan,
                "main_Am_np_rea": records[0].get("main_Am_np_rea", np.nan) if records else np.nan,
                "main_Am_wid_rea": records[0].get("main_Am_wid_rea", np.nan) if records else np.nan,
                "main_Delta_Am_acc": records[0].get("main_Delta_Am_acc", np.nan) if records else np.nan,
                "main_Delta_Am_don": records[0].get("main_Delta_Am_don", np.nan) if records else np.nan,
                "main_Delta_Am_np": records[0].get("main_Delta_Am_np", np.nan) if records else np.nan,
                "main_Delta_Am_wid": records[0].get("main_Delta_Am_wid", np.nan) if records else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_feature_dictionary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("main_Am_acc_neu", "Am_neutral", "neutral amine-side acceptor segmented term", "main aligned neutral object sigma integration"),
        ("main_Am_don_neu", "Am_neutral", "neutral amine-side donor segmented term", "main aligned neutral object sigma integration"),
        ("main_Am_np_neu", "Am_neutral", "neutral amine-side nonpolar segmented term", "main aligned neutral object sigma integration"),
        ("main_Am_wid_neu", "Am_neutral", "neutral amine-side width segmented term", "main aligned neutral object sigma integration"),
        ("main_Am_acc_rea", "Am_reacted", "reacted amine-side acceptor segmented term", "main routed reacted form sigma integration"),
        ("main_Am_don_rea", "Am_reacted", "reacted amine-side donor segmented term", "main routed reacted form sigma integration"),
        ("main_Am_np_rea", "Am_reacted", "reacted amine-side nonpolar segmented term", "main routed reacted form sigma integration"),
        ("main_Am_wid_rea", "Am_reacted", "reacted amine-side width segmented term", "main routed reacted form sigma integration"),
        ("main_Delta_Am_acc", "Delta_Am", "explicit reacted minus neutral acceptor transition", "main_Am_acc_rea - main_Am_acc_neu"),
        ("main_Delta_Am_don", "Delta_Am", "explicit reacted minus neutral donor transition", "main_Am_don_rea - main_Am_don_neu"),
        ("main_Delta_Am_np", "Delta_Am", "explicit reacted minus neutral nonpolar transition", "main_Am_np_rea - main_Am_np_neu"),
        ("main_Delta_Am_wid", "Delta_Am", "explicit reacted minus neutral width transition", "main_Am_wid_rea - main_Am_wid_neu"),
    ]
    for name, group, semantic, construction in specs:
        rows.append(
            {
                "feature_set_version": "routeA2_explicit_transition_phase1",
                "column_name": name,
                "feature_block": "routeA2_explicit_aligned_transition",
                "channel_scope": "main_only",
                "semantic_group": group,
                "semantic_description": semantic,
                "construction": construction,
                "segmentation_rule": f"donor<= {SIGMA_DONOR_THRESHOLD:.4f}; nonpolar between; acceptor>= {SIGMA_ACCEPTOR_THRESHOLD:.4f}; width=second_moment",
                "aq_nonaq_rule": "amine-side term defined in both AQ and dry rows when main reacted alignment exists",
                "routing_dependency": "reacted_channel_main",
                "not_old_routeA_summary_flag": 1,
            }
        )
    return pd.DataFrame(rows)


def build_distribution_summary(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
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
    rows: list[dict[str, object]] = []
    constructable = feature_df.loc[feature_df["explicit_transition_ready"] == 1].copy()
    for family in TARGET_FAMILIES:
        subset = constructable.loc[constructable["routeA2_family"] == family].copy()
        for col in feature_cols:
            series = subset[col].dropna().astype(float)
            if series.empty:
                continue
            rows.append(
                {
                    "feature_set_version": "routeA2_explicit_transition_phase1",
                    "routeA2_family": family,
                    "column_name": col,
                    "support_rows": int(series.shape[0]),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                    "min": float(series.min()),
                    "q25": float(series.quantile(0.25)),
                    "median": float(series.quantile(0.5)),
                    "q75": float(series.quantile(0.75)),
                    "max": float(series.max()),
                }
            )
    return pd.DataFrame(rows)


def build_coverage_summary(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in TARGET_FAMILIES:
        subset = feature_df.loc[feature_df["routeA2_family"] == family].copy()
        missing_rows = subset.loc[subset["explicit_transition_ready"] == 0].copy()
        rows.append(
            {
                "feature_set_version": "routeA2_explicit_transition_phase1",
                "routeA2_family": family,
                "support_rows": int(len(subset)),
                "explicit_transition_ready_rows": int(subset["explicit_transition_ready"].sum()),
                "explicit_transition_ready_rate": float(subset["explicit_transition_ready"].mean()),
                "missing_rows_due_to_reacted_coverage": int((subset["row_any_uncovered_reactive"] == 1).sum()),
                "missing_object_names_joined": "|".join(sorted(set(missing_rows["main_aligned_object_name"]))),
                "main_channel_protonation_row_rate": float((subset["main_reacted_channel_main"] == "protonation_base_assisted").mean()),
                "mean_old_routeA_main_delta_gap": float(subset["old_routeA_ra_routed_main_delta_gap_avg"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_report(feature_df: pd.DataFrame, coverage_df: pd.DataFrame, dist_df: pd.DataFrame) -> None:
    lines = [
        "# Route A2 Feature Audit Phase 1",
        "",
        "## Scope",
        "",
        "- Protocol: `strict_amine_object_holdout` only",
        "- System type: `single_amine_solvent` only",
        "- Families: `alcohol`, `sulfur_polar_aprotic`",
        "- No training or fitting is performed",
        "",
        "## Coverage",
        "",
    ]
    for row in coverage_df.itertuples(index=False):
        lines.append(
            f"- {row.routeA2_family}: n={int(row.support_rows)}, explicit-transition-ready rate {row.explicit_transition_ready_rate:.1%}, "
            f"missing reacted-coverage rows {int(row.missing_rows_due_to_reacted_coverage)}, missing objects `{row.missing_object_names_joined or 'none'}`."
        )
    lines.extend(["", "## Family Distribution Notes", ""])
    for family in TARGET_FAMILIES:
        fam = dist_df.loc[(dist_df["routeA2_family"] == family) & (dist_df["column_name"].str.startswith("main_Delta_"))].copy()
        fam = fam.sort_values("column_name")
        if fam.empty:
            continue
        joined = "; ".join(
            f"{r.column_name}: mean={r.mean:+.4f}, median={r.median:+.4f}, q25={r.q25:+.4f}, q75={r.q75:+.4f}"
            for r in fam.itertuples(index=False)
        )
        lines.append(f"- {family}: {joined}.")
    lines.extend(
        [
            "",
            "## Difference From Old Route A",
            "",
            "- Old Route A used compressed summary reacted deltas such as `delta_mean / delta_maxabs / delta_gap`.",
            "- This audit keeps explicit aligned neutral-state terms, reacted-state terms, and signed transition terms at row level.",
            "- Old Route A comparator columns are retained only as audit metadata, not as Route A2 features.",
        ]
    )
    (ROOT / "reports" / "routeA2_feature_audit_phase1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    mapping = build_canonical_map()
    descriptor_df = build_descriptor_table(mapping)
    alignment_df = build_alignment_audit(descriptor_df)
    feature_df = build_feature_table(alignment_df)
    feature_dict_df = build_feature_dictionary()
    dist_df = build_distribution_summary(feature_df)
    coverage_df = build_coverage_summary(feature_df)
    feature_df.to_csv(OUTPUTS_DIR / "routeA2_feature_table_phase1.csv", index=False)
    feature_dict_df.to_csv(OUTPUTS_DIR / "routeA2_feature_dictionary_phase1.csv", index=False)
    dist_df.to_csv(OUTPUTS_DIR / "routeA2_feature_distribution_summary_phase1.csv", index=False)
    coverage_df.to_csv(OUTPUTS_DIR / "routeA2_feature_coverage_summary_phase1.csv", index=False)
    write_report(feature_df, coverage_df, dist_df)


if __name__ == "__main__":
    main()
