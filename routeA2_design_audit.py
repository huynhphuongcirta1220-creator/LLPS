from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SIGMA_ACCEPTOR_THRESHOLD = 0.0084
SIGMA_DONOR_THRESHOLD = -0.0084
PRIMARY_PROTOCOL = "strict_amine_object_holdout"
CONFIRM_PROTOCOL = "ds_unknown"

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


def build_routeA2_coverage(alignment_df: pd.DataFrame) -> pd.DataFrame:
    route_df = pd.read_csv(OUTPUTS_DIR / "routeA_family_aware_increment_ready_table.csv").copy()
    route_df = route_df.loc[route_df["protocol"].isin([PRIMARY_PROTOCOL, CONFIRM_PROTOCOL])].copy()
    route_df["routeA2_family"] = route_df["solvent_family"].map(ROUTEA_FAMILY_MAP).fillna("other_organic_family")
    lookup = alignment_df.set_index("molecule_name").to_dict(orient="index")
    rows: list[dict[str, object]] = []

    for row in route_df.itertuples(index=False):
        amines = [str(row.amine1_name_canonical).strip()]
        if isinstance(row.amine2_name_canonical, str) and row.amine2_name_canonical.strip():
            amines.append(row.amine2_name_canonical.strip())
        records = [lookup.get(name, {}) for name in amines]
        rows.append(
            {
                "protocol": row.protocol,
                "routeA2_family": row.routeA2_family,
                "system_type": row.system_type,
                "row_id": row.row_id,
                "amine_objects_joined": "|".join(amines),
                "n_amine_objects": len(amines),
                "row_any_main_alignment_ready": int(any(int(rec.get("main_alignment_ready", 0)) == 1 for rec in records)),
                "row_all_main_alignment_ready": int(all(int(rec.get("main_alignment_ready", 0)) == 1 for rec in records)),
                "row_any_aux_alignment_ready": int(any(int(rec.get("aux_alignment_ready", 0)) == 1 for rec in records)),
                "row_any_bicarbonate_form": int(any(int(rec.get("has_bicarbonate_pair", 0)) == 1 for rec in records)),
                "row_any_carbamate_form": int(any(int(rec.get("has_carbamate_pair", 0)) == 1 for rec in records)),
                "row_any_nonreactive_bypass": int(any(str(rec.get("reactive_role", "")) == "nonreactive_solvent_like" for rec in records)),
                "row_any_uncovered_reactive": int(any(str(rec.get("reactive_role", "")) == "reactive_amine" and int(rec.get("main_alignment_ready", 0)) == 0 for rec in records)),
                "row_explicit_transition_ready": int(all(int(rec.get("explicit_transition_ready", 0)) == 1 for rec in records)),
                "row_main_channel_has_protonation": int(any(str(rec.get("reacted_channel_main", "")) == "protonation_base_assisted" for rec in records)),
            }
        )

    row_level = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for (protocol, family), subset in row_level.groupby(["protocol", "routeA2_family"], sort=False):
        objects = {
            obj
            for joined in subset["amine_objects_joined"]
            for obj in str(joined).split("|")
            if obj
        }
        obj_subset = alignment_df.loc[alignment_df["molecule_name"].isin(sorted(objects))].copy()
        summary_rows.append(
            {
                "audit_scope": "protocol_family",
                "protocol": protocol,
                "routeA2_family": family,
                "system_type": "all",
                "n_rows": int(len(subset)),
                "n_unique_row_id": int(subset["row_id"].nunique()),
                "n_distinct_amine_objects": int(len(objects)),
                "row_any_main_alignment_ready_rate": float(subset["row_any_main_alignment_ready"].mean()),
                "row_all_main_alignment_ready_rate": float(subset["row_all_main_alignment_ready"].mean()),
                "row_any_aux_alignment_ready_rate": float(subset["row_any_aux_alignment_ready"].mean()),
                "row_explicit_transition_ready_rate": float(subset["row_explicit_transition_ready"].mean()),
                "row_any_nonreactive_bypass_rate": float(subset["row_any_nonreactive_bypass"].mean()),
                "row_any_uncovered_reactive_rate": float(subset["row_any_uncovered_reactive"].mean()),
                "row_main_channel_has_protonation_rate": float(subset["row_main_channel_has_protonation"].mean()),
                "distinct_main_ready_objects": int(obj_subset["main_alignment_ready"].fillna(0).sum()),
                "distinct_aux_ready_objects": int(obj_subset["aux_alignment_ready"].fillna(0).sum()),
                "distinct_nonreactive_bypass_objects": int((obj_subset["reactive_role"] == "nonreactive_solvent_like").sum()),
                "observed_objects_joined": "|".join(sorted(objects)),
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    mapping = build_canonical_map()
    descriptor_df = build_descriptor_table(mapping)
    alignment_df = build_alignment_audit(descriptor_df)
    coverage_df = build_routeA2_coverage(alignment_df)
    alignment_df.to_csv(OUTPUTS_DIR / "routeA2_alignment_audit.csv", index=False)
    coverage_df.to_csv(OUTPUTS_DIR / "routeA2_coverage_audit.csv", index=False)


if __name__ == "__main__":
    main()
