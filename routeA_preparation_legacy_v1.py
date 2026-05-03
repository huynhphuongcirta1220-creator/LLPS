from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_SET_VERSION = "legacy_aligned_backbone_v1"
MODEL_NAME = "logreg"
PRIMARY_PROTOCOL = "strict_amine_object_holdout"
CONFIRM_PROTOCOL = "ds_unknown"
SIGMA_ACCEPTOR_THRESHOLD = 0.0084
SIGMA_DONOR_THRESHOLD = -0.0084

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = ROOT / "artifacts"

ROUTEA_FAMILY_MAP = {
    "alcohol": "alcohol",
    "sulfur_polar_aprotic": "sulfur_polar_aprotic",
    "glycol_ether_glyme": "glycol_ether_glyme",
}

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

BASIS_FEATURES = [
    "dGsolv_kcal_mol",
    "energy_dielectric",
    "area_total",
    "volume",
    "seg_acceptor",
    "seg_donor",
    "seg_nonpolar",
    "seg_width",
]


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
        parts = text.split("__", 1)
        return "neutral", parts[1], ""
    if text.startswith("reacted__"):
        parts = text.split("__", 2)
        return "reacted", parts[1], parts[2] if len(parts) > 2 else ""
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
    sigma_sq = np.square(sigma_axis)

    out = df[["job_name", "source_type"]].copy()
    out["seg_acceptor"] = (area_fraction * acceptor_mask).sum(axis=1)
    out["seg_donor"] = (area_fraction * donor_mask).sum(axis=1)
    out["seg_nonpolar"] = (area_fraction * nonpolar_mask).sum(axis=1)
    out["seg_width"] = (area_fraction * sigma_sq).sum(axis=1)
    return out


def build_descriptor_table(mapping: dict[str, str]) -> pd.DataFrame:
    surface = pd.read_csv(ROOT / "surface_descriptors_selected.csv").copy()
    sigma = pd.read_csv(ROOT / "sigma_profiles_wide_selected.csv").copy()
    sigma_seg = compute_sigma_segments(sigma)

    descriptor_df = surface.merge(sigma_seg, on=["job_name", "source_type"], how="inner", validate="one_to_one")
    parsed = descriptor_df["job_name"].map(parse_job_name).tolist()
    descriptor_df["descriptor_state"] = [item[0] for item in parsed]
    descriptor_df["molecule_name"] = [canonicalize_value(item[1], mapping) for item in parsed]
    descriptor_df["reacted_form"] = [item[2] for item in parsed]
    return descriptor_df


def robust_scales(neutral_df: pd.DataFrame) -> dict[str, float]:
    scales: dict[str, float] = {}
    for col in BASIS_FEATURES:
        series = neutral_df[col].astype(float)
        scale = float(series.quantile(0.75) - series.quantile(0.25))
        if scale <= 0:
            scale = float(series.std(ddof=1)) if series.std(ddof=1) > 0 else 1.0
        scales[col] = scale
    return scales


def routed_form_for_channel(channel: str, has_carbamate_pair: int, has_bicarbonate_pair: int) -> str:
    if channel == "carbamate_capable" and has_carbamate_pair:
        return "carbamate_pair"
    if channel == "protonation_base_assisted" and has_bicarbonate_pair:
        return "bicarbonate_pair"
    return ""


def summarize_delta(neutral_row: pd.Series, reacted_row: pd.Series, scales: dict[str, float], prefix: str) -> dict[str, float]:
    scaled_values = []
    out: dict[str, float] = {}
    for col in BASIS_FEATURES:
        raw_delta = float(reacted_row[col] - neutral_row[col])
        scaled_delta = raw_delta / scales[col]
        out[f"{prefix}_raw_delta__{col}"] = raw_delta
        out[f"{prefix}_scaled_delta__{col}"] = scaled_delta
        scaled_values.append(scaled_delta)
    vector = np.array(scaled_values, dtype=float)
    out[f"{prefix}_delta_mean"] = float(vector.mean())
    out[f"{prefix}_delta_maxabs"] = float(np.abs(vector).max())
    out[f"{prefix}_delta_gap"] = float(vector.max() - vector.min())
    return out


def build_object_summary(descriptor_df: pd.DataFrame) -> pd.DataFrame:
    role_df = pd.read_csv(ARTIFACTS_DIR / "reactive_role_dictionary.csv").copy()
    neutral_df = descriptor_df.loc[descriptor_df["descriptor_state"] == "neutral"].copy()
    reacted_df = descriptor_df.loc[descriptor_df["descriptor_state"] == "reacted"].copy()
    neutral_lookup = neutral_df.set_index("molecule_name")
    scales = robust_scales(neutral_df)

    rows: list[dict[str, object]] = []
    for role_row in role_df.itertuples(index=False):
        molecule = str(role_row.molecule_name)
        if molecule not in neutral_lookup.index:
            raise KeyError(f"Neutral descriptor missing for {molecule}")
        neutral_row = neutral_lookup.loc[molecule]
        if isinstance(neutral_row, pd.DataFrame):
            neutral_row = neutral_row.iloc[0]

        object_reacted = reacted_df.loc[reacted_df["molecule_name"] == molecule].copy()
        forms_present = sorted(set(object_reacted["reacted_form"]))
        has_carbamate_pair = int("carbamate_pair" in forms_present)
        has_bicarbonate_pair = int("bicarbonate_pair" in forms_present)
        reacted_form_count = int(len(forms_present))
        routed_main_form = routed_form_for_channel(str(role_row.reacted_channel_main), has_carbamate_pair, has_bicarbonate_pair)
        routed_aux_form = routed_form_for_channel(str(role_row.reacted_channel_aux), has_carbamate_pair, has_bicarbonate_pair)

        record: dict[str, object] = {
            "molecule_name": molecule,
            "source_roles_observed": role_row.source_roles_observed,
            "reactive_role": role_row.reactive_role,
            "reacted_channel_main": role_row.reacted_channel_main,
            "reacted_channel_aux": role_row.reacted_channel_aux,
            "routing_reason": role_row.routing_reason,
            "is_reacted_feature_eligible": int(role_row.is_reacted_feature_eligible),
            "notes": role_row.notes,
            "current_reacted_form_count_dictionary": int(role_row.current_reacted_form_count),
            "reacted_forms_observed_dictionary": role_row.current_reacted_forms_observed,
            "reacted_forms_observed_actual": "|".join(forms_present) if forms_present else "none",
            "reacted_form_count": reacted_form_count,
            "has_carbamate_pair": has_carbamate_pair,
            "has_bicarbonate_pair": has_bicarbonate_pair,
            "routed_main_form": routed_main_form or "none",
            "routed_aux_form": routed_aux_form or "none",
            "routed_main_available": int(bool(routed_main_form)),
            "routed_aux_available": int(bool(routed_aux_form)),
        }

        for col in BASIS_FEATURES:
            record[f"neutral__{col}"] = float(neutral_row[col])

        for form_name in ["bicarbonate_pair", "carbamate_pair"]:
            form_row = object_reacted.loc[object_reacted["reacted_form"] == form_name]
            prefix = form_name.replace("_pair", "")
            record[f"{prefix}_available"] = int(not form_row.empty)
            if form_row.empty:
                for col in BASIS_FEATURES:
                    record[f"{prefix}_raw_delta__{col}"] = np.nan
                    record[f"{prefix}_scaled_delta__{col}"] = np.nan
                record[f"{prefix}_delta_mean"] = np.nan
                record[f"{prefix}_delta_maxabs"] = np.nan
                record[f"{prefix}_delta_gap"] = np.nan
            else:
                form_series = form_row.iloc[0]
                record.update(summarize_delta(neutral_row, form_series, scales, prefix))

        for branch_name, form_name in [("routed_main", routed_main_form), ("routed_aux", routed_aux_form)]:
            if not form_name:
                record[f"{branch_name}_delta_mean"] = np.nan
                record[f"{branch_name}_delta_maxabs"] = np.nan
                record[f"{branch_name}_delta_gap"] = np.nan
            else:
                prefix = form_name.replace("_pair", "")
                record[f"{branch_name}_delta_mean"] = record[f"{prefix}_delta_mean"]
                record[f"{branch_name}_delta_maxabs"] = record[f"{prefix}_delta_maxabs"]
                record[f"{branch_name}_delta_gap"] = record[f"{prefix}_delta_gap"]

        record["routeA_object_summary_ready_flag"] = int(
            record["is_reacted_feature_eligible"] == 1 and record["routed_main_available"] == 1
        )
        rows.append(record)

    return pd.DataFrame(rows)


def load_routeA_rows() -> pd.DataFrame:
    rows_df = pd.read_csv(OUTPUTS_DIR / "residual_diagnosis_rows_legacy_v1_logreg.csv").copy()
    route_df = rows_df.loc[rows_df["organic_solvent_flag"] == 1].copy()
    route_df["routeA_family"] = route_df["solvent_family"].map(ROUTEA_FAMILY_MAP).fillna("other_organic_family")
    route_df["routeA_focus_family_flag"] = route_df["routeA_family"].isin(
        ["alcohol", "sulfur_polar_aprotic", "glycol_ether_glyme"]
    ).astype(int)
    return route_df


def object_lookup(summary_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    return summary_df.set_index("molecule_name").to_dict(orient="index")


def safe_mean(values: list[float]) -> float:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return np.nan
    return float(np.mean(clean))


def build_row_candidates(route_rows: pd.DataFrame, object_summary_df: pd.DataFrame) -> pd.DataFrame:
    lookup = object_lookup(object_summary_df)
    candidate_rows: list[dict[str, object]] = []

    for row in route_rows.itertuples(index=False):
        amines = [str(row.amine1_name_canonical).strip()]
        if isinstance(row.amine2_name_canonical, str) and row.amine2_name_canonical.strip():
            amines.append(row.amine2_name_canonical.strip())

        object_records = [lookup.get(amine, {}) for amine in amines]
        eligible_records = [rec for rec in object_records if rec and int(rec.get("is_reacted_feature_eligible", 0)) == 1]
        main_records = [rec for rec in eligible_records if int(rec.get("routed_main_available", 0)) == 1]
        aux_records = [rec for rec in eligible_records if int(rec.get("routed_aux_available", 0)) == 1]

        record: dict[str, object] = {
            "row_id": row.row_id,
            "feature_set_version": FEATURE_SET_VERSION,
            "model_name": MODEL_NAME,
            "protocol": row.protocol,
            "fold_id": row.fold_id,
            "routeA_family": row.routeA_family,
            "routeA_focus_family_flag": row.routeA_focus_family_flag,
            "system_type": row.system_type,
            "solvent_name_canonical": row.solvent_name_canonical,
            "solvent_family": row.solvent_family,
            "amine1_name_canonical": row.amine1_name_canonical,
            "amine2_name_canonical": row.amine2_name_canonical,
            "y_true": row.y_true,
            "p_logreg": row.p_logreg,
            "residual": row.residual,
            "abs_residual": row.abs_residual,
            "error_flag": row.error_flag,
            "official_primary_evidence_row_flag": row.official_primary_evidence_row_flag,
            "ds_unknown_confirmation_row_flag": row.ds_unknown_confirmation_row_flag,
            "ra_object_count": len(amines),
            "ra_eligible_object_count": len(eligible_records),
            "ra_ineligible_object_count": len(amines) - len(eligible_records),
            "ra_routed_main_available_count": len(main_records),
            "ra_routed_aux_available_count": len(aux_records),
            "ra_has_carbamate_pair_any": int(any(int(rec.get("has_carbamate_pair", 0)) == 1 for rec in eligible_records)),
            "ra_has_bicarbonate_pair_any": int(any(int(rec.get("has_bicarbonate_pair", 0)) == 1 for rec in eligible_records)),
            "ra_reacted_form_count_avg": safe_mean([rec.get("reacted_form_count", np.nan) for rec in eligible_records]),
            "ra_main_channel_carbamate_object_count": int(
                sum(str(rec.get("reacted_channel_main", "")) == "carbamate_capable" for rec in eligible_records)
            ),
            "ra_main_channel_protonation_object_count": int(
                sum(str(rec.get("reacted_channel_main", "")) == "protonation_base_assisted" for rec in eligible_records)
            ),
            "ra_row_increment_eligible_flag": int(len(eligible_records) == len(amines) and len(main_records) >= 1),
            "ra_skipped_object_names": "|".join(
                [amine for amine, rec in zip(amines, object_records) if not rec or int(rec.get("is_reacted_feature_eligible", 0)) == 0]
            ),
        }

        for prefix, records in [("ra_routed_main", main_records), ("ra_routed_aux", aux_records)]:
            record[f"{prefix}_delta_mean_avg"] = safe_mean([rec.get(f"{prefix[3:]}_delta_mean", np.nan) for rec in records])
            record[f"{prefix}_delta_maxabs_avg"] = safe_mean([rec.get(f"{prefix[3:]}_delta_maxabs", np.nan) for rec in records])
            record[f"{prefix}_delta_gap_avg"] = safe_mean([rec.get(f"{prefix[3:]}_delta_gap", np.nan) for rec in records])

        candidate_rows.append(record)

    return pd.DataFrame(candidate_rows)


def build_increment_ready_table(candidate_df: pd.DataFrame) -> pd.DataFrame:
    df = candidate_df.copy()
    family_labels = ["alcohol", "sulfur_polar_aprotic", "glycol_ether_glyme", "other_organic_family"]
    shared_features = [
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

    for family in family_labels:
        flag_col = f"fa_is_{family}"
        df[flag_col] = (df["routeA_family"] == family).astype(int)
        for feature in shared_features:
            gated_col = f"fa_{family}__{feature}"
            df[gated_col] = np.where(df["routeA_family"] == family, df[feature].fillna(0.0), 0.0)

    return df


def build_coverage_audit(candidate_df: pd.DataFrame, object_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (protocol, family), subset in candidate_df.groupby(["protocol", "routeA_family"], sort=False):
        object_names = {
            str(name).strip()
            for name in list(subset["amine1_name_canonical"]) + list(subset["amine2_name_canonical"])
            if pd.notna(name) and str(name).strip()
        }
        eligible_object_names = {
            name
            for name in object_names
            if not object_summary_df.loc[object_summary_df["molecule_name"] == name].empty
            and int(object_summary_df.loc[object_summary_df["molecule_name"] == name, "is_reacted_feature_eligible"].iloc[0]) == 1
        }
        family_objects = object_summary_df.loc[object_summary_df["molecule_name"].isin(sorted(object_names))].copy()
        rows.append(
            {
                "audit_scope": "protocol_family",
                "protocol": protocol,
                "routeA_family": family,
                "system_type": "all",
                "amine_object": "",
                "n_rows": int(len(subset)),
                "n_unique_row_id": int(subset["row_id"].nunique()),
                "n_distinct_amine_objects": int(len(object_names)),
                "n_distinct_eligible_amine_objects": int(len(eligible_object_names)),
                "row_increment_eligible_rate": float(subset["ra_row_increment_eligible_flag"].mean()),
                "row_any_eligible_rate": float((subset["ra_eligible_object_count"] >= 1).mean()),
                "row_main_available_rate": float((subset["ra_routed_main_available_count"] >= 1).mean()),
                "row_aux_available_rate": float((subset["ra_routed_aux_available_count"] >= 1).mean()),
                "row_has_carbamate_pair_rate": float(subset["ra_has_carbamate_pair_any"].mean()),
                "row_has_bicarbonate_pair_rate": float(subset["ra_has_bicarbonate_pair_any"].mean()),
                "mean_reacted_form_count": float(subset["ra_reacted_form_count_avg"].mean()),
                "main_carbamate_object_share": float((family_objects["reacted_channel_main"] == "carbamate_capable").mean()) if not family_objects.empty else np.nan,
                "main_protonation_object_share": float((family_objects["reacted_channel_main"] == "protonation_base_assisted").mean()) if not family_objects.empty else np.nan,
                "observed_objects_joined": "|".join(sorted(object_names)),
            }
        )

    for (protocol, family, system_type), subset in candidate_df.groupby(["protocol", "routeA_family", "system_type"], sort=False):
        subset_object_names = {
            str(name).strip()
            for name in list(subset["amine1_name_canonical"]) + list(subset["amine2_name_canonical"])
            if pd.notna(name) and str(name).strip()
        }
        subset_eligible_names = {
            name
            for name in subset_object_names
            if not object_summary_df.loc[object_summary_df["molecule_name"] == name].empty
            and int(object_summary_df.loc[object_summary_df["molecule_name"] == name, "is_reacted_feature_eligible"].iloc[0]) == 1
        }
        rows.append(
            {
                "audit_scope": "protocol_family_system_type",
                "protocol": protocol,
                "routeA_family": family,
                "system_type": system_type,
                "amine_object": "",
                "n_rows": int(len(subset)),
                "n_unique_row_id": int(subset["row_id"].nunique()),
                "n_distinct_amine_objects": int(len(subset_object_names)),
                "n_distinct_eligible_amine_objects": int(len(subset_eligible_names)),
                "row_increment_eligible_rate": float(subset["ra_row_increment_eligible_flag"].mean()),
                "row_any_eligible_rate": float((subset["ra_eligible_object_count"] >= 1).mean()),
                "row_main_available_rate": float((subset["ra_routed_main_available_count"] >= 1).mean()),
                "row_aux_available_rate": float((subset["ra_routed_aux_available_count"] >= 1).mean()),
                "row_has_carbamate_pair_rate": float(subset["ra_has_carbamate_pair_any"].mean()),
                "row_has_bicarbonate_pair_rate": float(subset["ra_has_bicarbonate_pair_any"].mean()),
                "mean_reacted_form_count": float(subset["ra_reacted_form_count_avg"].mean()),
                "main_carbamate_object_share": np.nan,
                "main_protonation_object_share": np.nan,
                "observed_objects_joined": "",
            }
        )

    object_records = []
    for row in candidate_df.itertuples(index=False):
        for amine in [row.amine1_name_canonical, row.amine2_name_canonical]:
            if isinstance(amine, str) and amine:
                object_records.append(
                    {
                        "protocol": row.protocol,
                        "routeA_family": row.routeA_family,
                        "amine_object": amine,
                        "row_id": row.row_id,
                    }
                )
    object_usage = pd.DataFrame(object_records)
    if not object_usage.empty:
        usage_summary = object_usage.groupby(["protocol", "routeA_family", "amine_object"], sort=False).agg(n_rows=("row_id", "nunique")).reset_index()
        usage_summary = usage_summary.merge(
            object_summary_df[
                [
                    "molecule_name",
                    "is_reacted_feature_eligible",
                    "reacted_channel_main",
                    "reacted_channel_aux",
                    "has_carbamate_pair",
                    "has_bicarbonate_pair",
                    "reacted_form_count",
                ]
            ],
            left_on="amine_object",
            right_on="molecule_name",
            how="left",
            validate="many_to_one",
        )
        for row in usage_summary.itertuples(index=False):
            rows.append(
                {
                    "audit_scope": "protocol_family_amine_object",
                    "protocol": row.protocol,
                    "routeA_family": row.routeA_family,
                    "system_type": "",
                    "amine_object": row.amine_object,
                    "n_rows": int(row.n_rows),
                    "n_unique_row_id": int(row.n_rows),
                    "n_distinct_amine_objects": 1,
                    "n_distinct_eligible_amine_objects": int(row.is_reacted_feature_eligible) if pd.notna(row.is_reacted_feature_eligible) else 0,
                    "row_increment_eligible_rate": float(row.is_reacted_feature_eligible) if pd.notna(row.is_reacted_feature_eligible) else 0.0,
                    "row_any_eligible_rate": float(row.is_reacted_feature_eligible) if pd.notna(row.is_reacted_feature_eligible) else 0.0,
                    "row_main_available_rate": np.nan,
                    "row_aux_available_rate": np.nan,
                    "row_has_carbamate_pair_rate": float(row.has_carbamate_pair) if pd.notna(row.has_carbamate_pair) else np.nan,
                    "row_has_bicarbonate_pair_rate": float(row.has_bicarbonate_pair) if pd.notna(row.has_bicarbonate_pair) else np.nan,
                    "mean_reacted_form_count": float(row.reacted_form_count) if pd.notna(row.reacted_form_count) else np.nan,
                    "main_carbamate_object_share": float(str(row.reacted_channel_main) == "carbamate_capable"),
                    "main_protonation_object_share": float(str(row.reacted_channel_main) == "protonation_base_assisted"),
                    "observed_objects_joined": row.amine_object,
                }
            )

    return pd.DataFrame(rows)


def focus_family_objects(audit_df: pd.DataFrame, protocol: str, family: str) -> str:
    subset = audit_df.loc[
        (audit_df["audit_scope"] == "protocol_family_amine_object")
        & (audit_df["protocol"] == protocol)
        & (audit_df["routeA_family"] == family)
    ].sort_values(["n_rows", "amine_object"], ascending=[False, True]).head(6)
    if subset.empty:
        return "none"
    return "; ".join(f"{row.amine_object}(n={int(row.n_rows)})" for row in subset.itertuples(index=False))


def write_routeA_report(candidate_df: pd.DataFrame, object_summary_df: pd.DataFrame, audit_df: pd.DataFrame) -> None:
    family_audit = audit_df.loc[audit_df["audit_scope"] == "protocol_family"].copy()
    strict_family = family_audit.loc[family_audit["protocol"] == PRIMARY_PROTOCOL].copy()
    strict_family = strict_family.sort_values("routeA_family")

    ineligible_objects = object_summary_df.loc[
        (object_summary_df["is_reacted_feature_eligible"] == 0)
        & (object_summary_df["molecule_name"].isin(set(candidate_df["amine1_name_canonical"]) | set(candidate_df["amine2_name_canonical"])))
    ]["molecule_name"].tolist()

    lines = [
        "# Route A Preparation Legacy V1",
        "",
        "## Scope",
        "",
        "- Baseline remains frozen: `legacy_aligned_backbone_v1 + Logistic Regression`.",
        "- This step prepares Route A reacted-delta inputs only. No residual fitting is launched here.",
        "- Route scope is `organic_solvent_containing`, with `single_amine_solvent` as the main training-ready body.",
        "",
        "## Coverage Summary",
        "",
    ]
    for row in strict_family.itertuples(index=False):
        lines.append(
            f"- {row.routeA_family}: n={int(row.n_rows)}, increment-eligible row rate {row.row_increment_eligible_rate:.1%}, "
            f"distinct eligible objects {int(row.n_distinct_eligible_amine_objects)}, form count mean {row.mean_reacted_form_count:.2f}, "
            f"top objects {focus_family_objects(audit_df, PRIMARY_PROTOCOL, row.routeA_family)}."
        )

    lines.extend(
        [
            "",
            "## Form-Structure Notes",
            "",
            "- `alcohol` and `sulfur_polar_aprotic` are the main Route A preparation families and both are dominated by carbamate-capable amine objects with good reacted coverage.",
            "- `glycol_ether_glyme` stays separate because residual direction was weaker and partly opposite in prior diagnosis; it should not be merged into one generic organic increment branch.",
            "- `other_organic_family` is retained only for audit completeness, not as a first-line residual-training target.",
            "",
            "## Current Exclusions",
            "",
            f"- Route A row-level reacted coverage misses only a small set of rows driven by uncovered object(s): {', '.join(sorted(ineligible_objects)) if ineligible_objects else 'none'}.",
            "- `dual_amine_solvent` rows are kept in the preparation table but should not anchor the first residual fit because the subgroup is tiny and class-degenerate.",
        ]
    )

    (REPORTS_DIR / "routeA_preparation_legacy_v1.md").write_text("\n".join(lines), encoding="utf-8")


def write_feature_scope(candidate_df: pd.DataFrame) -> None:
    shared_features = [
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
    family_features = [col for col in candidate_df.columns if col.startswith("fa_")]

    lines = [
        "# Route A Feature Scope",
        "",
        "## Object-Level Summary Definition",
        "",
        "- Neutral/reacted deltas are computed on a fixed descriptor basis:",
        "- `dGsolv_kcal_mol`, `energy_dielectric`, `area_total`, `volume`, `seg_acceptor`, `seg_donor`, `seg_nonpolar`, `seg_width`.",
        "- Each basis delta is converted to a dimensionless scaled delta using the neutral-object robust scale (IQR fallback to standard deviation).",
        "- Object-level summary metrics are then:",
        "- `delta_mean`: mean of the scaled delta vector.",
        "- `delta_maxabs`: maximum absolute scaled delta.",
        "- `delta_gap`: max scaled delta minus min scaled delta.",
        "",
        "## Routing Constraint",
        "",
        "- `nonreactive_solvent_like` objects bypass reacted amine feature generation.",
        "- `carbamate_capable` main routing uses `carbamate_pair` when available.",
        "- `protonation_base_assisted` routing uses `bicarbonate_pair` when available.",
        "- Observed forms are still audited separately through `has_carbamate_pair`, `has_bicarbonate_pair`, and `reacted_form_count`.",
        "",
        "## Shared Route A Candidate Features",
        "",
    ]
    lines.extend([f"- `{feature}`" for feature in shared_features])
    lines.extend(
        [
            "",
            "## Family-Aware Gated Features",
            "",
            "- The increment-ready table contains explicit family labels plus gated feature copies for:",
            "- `alcohol`",
            "- `sulfur_polar_aprotic`",
            "- `glycol_ether_glyme`",
            "- `other_organic_family`",
            "- These gated copies are preparation artifacts only; they do not change the frozen baseline.",
            "",
            "## Explicitly Out Of Scope",
            "",
            "- reacted absolute descriptor blocks",
            "- raw sigma bins",
            "- any baseline backbone modification",
            "- Route B object-score work",
            "- any final residual model fitting or model sweep",
            "",
            "## Minimal Future Route A Fit",
            "",
            "- Start with `strict_amine_object_holdout` only, then repeat on `DS-unknown` as confirmation.",
            "- First residual-fit target should be `single_amine_solvent` rows within `alcohol` and `sulfur_polar_aprotic`, trained as separate family branches.",
            "- Keep `glycol_ether_glyme` as a separate diagnostic branch because its bias direction differed from the main organic families.",
            "- Use the frozen baseline probability `p_logreg` and the low-dimensional reacted delta candidates only.",
            "- Recommended first residual learner: a simple linear residual corrector on `y_true - p_logreg`, evaluated fold-safely within the existing outer-fold protocol.",
        ]
    )

    (REPORTS_DIR / "routeA_feature_scope.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    mapping = build_canonical_map()
    descriptor_df = build_descriptor_table(mapping)
    object_summary_df = build_object_summary(descriptor_df)
    route_rows = load_routeA_rows()
    candidate_df = build_row_candidates(route_rows, object_summary_df)
    increment_ready_df = build_increment_ready_table(candidate_df)
    audit_df = build_coverage_audit(candidate_df, object_summary_df)

    object_summary_df.to_csv(OUTPUTS_DIR / "routeA_reacted_object_summary.csv", index=False)
    candidate_df.to_csv(OUTPUTS_DIR / "routeA_reacted_delta_candidates.csv", index=False)
    increment_ready_df.to_csv(OUTPUTS_DIR / "routeA_family_aware_increment_ready_table.csv", index=False)
    audit_df.to_csv(OUTPUTS_DIR / "routeA_coverage_audit.csv", index=False)

    write_routeA_report(candidate_df, object_summary_df, audit_df)
    write_feature_scope(increment_ready_df)


if __name__ == "__main__":
    main()
