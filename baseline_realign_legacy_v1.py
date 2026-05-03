from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_SET_VERSION = "legacy_aligned_backbone_v1"
WATER_MW = 18.01528
SIGMA_ACCEPTOR_THRESHOLD = 0.0084
SIGMA_DONOR_THRESHOLD = -0.0084

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
ARTIFACTS_DIR = ROOT / "artifacts"


@dataclass
class SlotComponent:
    name: str
    weight_pct: float
    molecular_weight: float
    profile_area: np.ndarray


def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)


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


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(val):
        return default
    return val


def safe_log(value: float, eps: float = 1e-12) -> float:
    return float(np.log(max(value, eps)))


def safe_divide(num: float, denom: float, default: float = 0.0) -> float:
    if denom == 0 or np.isnan(denom):
        return default
    return float(num / denom)


def load_main_table(mapping: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "final_modeling_dataset.csv").copy()
    df.insert(0, "row_id", [f"R{i:06d}" for i in range(1, len(df) + 1)])
    df["feature_set_version"] = FEATURE_SET_VERSION
    df["amine1_name_canonical"] = df["Amine 1"].map(lambda x: canonicalize_value(x, mapping))
    df["amine2_name_canonical"] = df["Amine 2"].map(lambda x: canonicalize_value(x, mapping))
    df["solvent_name_canonical"] = df["Solvent"].map(lambda x: canonicalize_value(x, mapping))
    df["water_name_canonical"] = "H2O"
    df["amine_set_key"] = df.apply(
        lambda row: "|".join(sorted([x for x in [row["amine1_name_canonical"], row["amine2_name_canonical"]] if x])),
        axis=1,
    )
    return df


def validate_reactive_dictionary(main_df: pd.DataFrame) -> None:
    reactive_dict = pd.read_csv(ARTIFACTS_DIR / "reactive_role_dictionary.csv")
    known = set(reactive_dict["molecule_name"].astype(str).str.strip())
    amine_like = set(main_df["amine1_name_canonical"]) | set(main_df["amine2_name_canonical"])
    missing = sorted(x for x in amine_like if x and x not in known)
    if missing:
        raise ValueError(f"Amine-like objects missing from reactive role dictionary: {missing}")


def build_neutral_object_table(mapping: dict[str, str]) -> tuple[pd.DataFrame, np.ndarray]:
    surface = pd.read_csv(ROOT / "surface_descriptors_selected.csv")
    sigma = pd.read_csv(ROOT / "sigma_profiles_wide_selected.csv")

    surface = surface.loc[surface["source_type"] == "neutral"].copy()
    sigma = sigma.loc[sigma["source_type"] == "neutral"].copy()
    surface["molecule_name_canonical"] = surface["job_name"].map(parse_base_name).map(lambda x: canonicalize_value(x, mapping))
    sigma["molecule_name_canonical"] = sigma["job_name"].map(parse_base_name).map(lambda x: canonicalize_value(x, mapping))

    sigma_axis = sigma[[f"sigma_{i}" for i in range(1, 61)]].iloc[0].astype(float).to_numpy()
    area_cols = [f"area_{i}" for i in range(1, 61)]
    sigma_area = sigma[["molecule_name_canonical"] + area_cols].copy()
    merged = surface.merge(sigma_area, on="molecule_name_canonical", how="inner", validate="one_to_one")

    area_matrix = merged[area_cols].astype(float).to_numpy()
    total_area = area_matrix.sum(axis=1)
    area_fraction = np.divide(area_matrix, total_area[:, None], out=np.zeros_like(area_matrix), where=total_area[:, None] > 0)

    acceptor_mask = sigma_axis >= SIGMA_ACCEPTOR_THRESHOLD
    donor_mask = sigma_axis <= SIGMA_DONOR_THRESHOLD
    nonpolar_mask = (~acceptor_mask) & (~donor_mask)
    sigma_sq = np.square(sigma_axis)

    merged["seg_acceptor"] = (area_fraction * acceptor_mask).sum(axis=1)
    merged["seg_donor"] = (area_fraction * donor_mask).sum(axis=1)
    merged["seg_nonpolar"] = (area_fraction * nonpolar_mask).sum(axis=1)
    merged["seg_width"] = (area_fraction * sigma_sq).sum(axis=1)

    keep_cols = [
        "molecule_name_canonical",
        "area_total",
        "seg_acceptor",
        "seg_donor",
        "seg_nonpolar",
        "seg_width",
    ] + area_cols
    return merged[keep_cols].copy(), sigma_axis


def object_table_to_lookup(object_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    area_cols = [col for col in object_df.columns if col.startswith("area_") and col != "area_total"]
    lookup: dict[str, dict[str, object]] = {}
    for row in object_df.itertuples(index=False):
        data = {
            "area_total": float(row.area_total),
            "seg_acceptor": float(row.seg_acceptor),
            "seg_donor": float(row.seg_donor),
            "seg_nonpolar": float(row.seg_nonpolar),
            "seg_width": float(row.seg_width),
            "area_profile": np.array([float(getattr(row, col)) for col in area_cols], dtype=float),
        }
        lookup[str(row.molecule_name_canonical)] = data
    return lookup


def build_slot_component(name: str, weight_pct: float, molecular_weight: float, object_lookup: dict[str, dict[str, object]]) -> SlotComponent | None:
    if not name or weight_pct <= 0 or molecular_weight <= 0:
        return None
    if name not in object_lookup:
        raise KeyError(f"Neutral descriptor missing for {name}")
    return SlotComponent(
        name=name,
        weight_pct=weight_pct,
        molecular_weight=molecular_weight,
        profile_area=np.array(object_lookup[name]["area_profile"], dtype=float),
    )


def component_moles(component: SlotComponent | None) -> float:
    if component is None:
        return 0.0
    return safe_divide(component.weight_pct, component.molecular_weight, default=0.0)


def mix_area_profiles(components: list[SlotComponent | None]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    weights: list[float] = []
    for component in components:
        if component is None:
            continue
        mole = component_moles(component)
        if mole <= 0:
            continue
        vectors.append(component.profile_area)
        weights.append(mole)
    if not vectors:
        return np.zeros(60, dtype=float)
    mixed_area = np.sum(np.stack(vectors, axis=0) * np.array(weights)[:, None], axis=0)
    total = mixed_area.sum()
    if total <= 0:
        return np.zeros(60, dtype=float)
    return mixed_area / total


def profile_to_segments(profile: np.ndarray, sigma_axis: np.ndarray) -> dict[str, float]:
    acceptor_mask = sigma_axis >= SIGMA_ACCEPTOR_THRESHOLD
    donor_mask = sigma_axis <= SIGMA_DONOR_THRESHOLD
    nonpolar_mask = (~acceptor_mask) & (~donor_mask)
    return {
        "acceptor": float(profile[acceptor_mask].sum()),
        "donor": float(profile[donor_mask].sum()),
        "nonpolar": float(profile[nonpolar_mask].sum()),
        "width": float(np.sum(profile * np.square(sigma_axis))),
    }


def effective_component_mw(weight_total: float, mole_total: float, fallback: float) -> float:
    if mole_total > 0 and weight_total > 0:
        return safe_divide(weight_total, mole_total, default=fallback)
    return fallback


def build_legacy_aligned_table(main_df: pd.DataFrame, object_lookup: dict[str, dict[str, object]], sigma_axis: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: list[dict[str, object]] = []
    modeling_rows: list[dict[str, object]] = []

    feature_specs = [
        ("x_Am1", "L0", "Legacy-aligned mole fraction of amine 1", "main wt% + molecular weights", "legacy_L0"),
        ("x_Am2", "L0", "Legacy-aligned mole fraction of amine 2", "main wt% + molecular weights", "legacy_L0"),
        ("x_Sol", "L0", "Legacy-aligned mole fraction of solvent", "main wt% + molecular weights", "legacy_L0"),
        ("x_H2O", "L0", "Legacy-aligned mole fraction of water", "main wt% + molecular weights", "legacy_L0"),
        ("S_ideal", "L1", "Signed ideal mixing entropy sum(x ln x)", "L0 mole fractions", "legacy_L1"),
        ("TS_mix", "L1", "Temperature times signed ideal mixing entropy", "S_ideal + temperature", "legacy_L1"),
        ("inv_T", "L1", "1000 / T legacy scaling", "Temperature(K)", "legacy_L1"),
        ("xA_invT", "L1", "Total amine mole fraction times 1000 / T", "x_Am1 + x_Am2 + temperature", "legacy_L1"),
        ("ln_MW_rat", "L2", "Log ratio of effective amine MW to medium reference MW", "component weights + MWs", "legacy_L2"),
        ("Am_Sol_rat", "L2", "Total amine mole fraction over total medium mole fraction", "L0 mole fractions", "legacy_L2"),
        ("Water_ent", "L2", "Signed water entropy term with AQ/NONAQ branch", "x_H2O", "legacy_L2"),
        ("Am_acc", "L3", "Amine-mixture acceptor integrated sigma area", "neutral sigma profile integration", "legacy_L3"),
        ("Am_don", "L3", "Amine-mixture donor integrated sigma area", "neutral sigma profile integration", "legacy_L3"),
        ("Am_np", "L3", "Amine-mixture nonpolar integrated sigma area", "neutral sigma profile integration", "legacy_L3"),
        ("Am_wid", "L3", "Amine-mixture sigma second-moment width", "neutral sigma profile integration", "legacy_L3"),
        ("Sv_acc", "L3", "Medium-mixture acceptor integrated sigma area", "neutral sigma profile integration", "legacy_L3"),
        ("Sv_don", "L3", "Medium-mixture donor integrated sigma area", "neutral sigma profile integration", "legacy_L3"),
        ("Sv_np", "L3", "Medium-mixture nonpolar integrated sigma area", "neutral sigma profile integration", "legacy_L3"),
        ("Sv_wid", "L3", "Medium-mixture sigma second-moment width", "neutral sigma profile integration", "legacy_L3"),
        ("Delta_S_np", "L3", "Absolute Am_np minus Sv_np mismatch", "Am_np + Sv_np", "legacy_L3"),
        ("Delta_S_acc", "L3", "Absolute Am_acc minus Sv_acc mismatch", "Am_acc + Sv_acc", "legacy_L3"),
        ("Sal_Out", "L3", "AQ-only salt-out proxy: x_H2O times solvent nonpolar segment", "x_H2O + solvent sigma nonpolar", "legacy_L3"),
    ]

    for _, row in main_df.iterrows():
        wt1 = safe_float(row["Amine 1 wt%"])
        wt2 = safe_float(row["Amine 2 wt%"])
        wts = safe_float(row["Solvent wt%"])
        wth = safe_float(row["H2O wt%"])
        mw1 = safe_float(row["Amine1_MW"], default=0.0)
        mw2 = safe_float(row["Amine2_MW"], default=0.0)
        mws = safe_float(row["Solv_MW"], default=0.0)
        T = safe_float(row["Temperature(K)"], default=313.0)

        amine1 = build_slot_component(row["amine1_name_canonical"], wt1, mw1, object_lookup)
        amine2 = build_slot_component(row["amine2_name_canonical"], wt2, mw2, object_lookup)
        solvent = build_slot_component(row["solvent_name_canonical"], wts, mws, object_lookup)
        water = build_slot_component(row["water_name_canonical"], wth, WATER_MW, object_lookup) if wth > 0 else None

        m1 = component_moles(amine1)
        m2 = component_moles(amine2)
        ms = component_moles(solvent)
        mh = component_moles(water)
        mt = m1 + m2 + ms + mh
        if mt <= 0:
            continue

        x_am1 = m1 / mt
        x_am2 = m2 / mt
        x_sol = ms / mt
        x_h2o = mh / mt
        x_amine_total = x_am1 + x_am2
        x_medium_total = x_sol + x_h2o

        s_ideal = sum(x * safe_log(x) for x in [x_am1, x_am2, x_sol, x_h2o] if x > 0)
        inv_t = 1000.0 / T

        effective_amine_mw = effective_component_mw(wt1 + wt2, m1 + m2, fallback=max(mw1, 1.0))
        if wts > 0 and mws > 0:
            medium_ref_mw = mws
        elif wth > 0:
            medium_ref_mw = WATER_MW
        else:
            medium_ref_mw = max(mws, WATER_MW)

        ln_mw_rat = safe_log(safe_divide(effective_amine_mw, medium_ref_mw, default=1.0))
        am_sol_rat = safe_divide(x_amine_total, x_medium_total, default=0.0)
        water_ent = x_h2o * safe_log(x_h2o) if x_h2o > 0 else 0.0

        am_profile = mix_area_profiles([amine1, amine2])
        sv_profile = mix_area_profiles([solvent, water])
        am_seg = profile_to_segments(am_profile, sigma_axis)
        sv_seg = profile_to_segments(sv_profile, sigma_axis)

        if wth > 0 and wts > 0 and solvent is not None:
            solvent_profile = mix_area_profiles([solvent])
            solvent_seg = profile_to_segments(solvent_profile, sigma_axis)
            sal_out = x_h2o * solvent_seg["nonpolar"]
        else:
            sal_out = 0.0

        feature_values = {
            "x_Am1": x_am1,
            "x_Am2": x_am2,
            "x_Sol": x_sol,
            "x_H2O": x_h2o,
            "S_ideal": s_ideal,
            "TS_mix": T * s_ideal,
            "inv_T": inv_t,
            "xA_invT": x_amine_total * inv_t,
            "ln_MW_rat": ln_mw_rat,
            "Am_Sol_rat": am_sol_rat,
            "Water_ent": water_ent,
            "Am_acc": am_seg["acceptor"],
            "Am_don": am_seg["donor"],
            "Am_np": am_seg["nonpolar"],
            "Am_wid": am_seg["width"],
            "Sv_acc": sv_seg["acceptor"],
            "Sv_don": sv_seg["donor"],
            "Sv_np": sv_seg["nonpolar"],
            "Sv_wid": sv_seg["width"],
            "Delta_S_np": abs(am_seg["nonpolar"] - sv_seg["nonpolar"]),
            "Delta_S_acc": abs(am_seg["acceptor"] - sv_seg["acceptor"]),
            "Sal_Out": sal_out,
        }

        modeling_rows.append(
            {
                "row_id": row["row_id"],
                "feature_set_version": FEATURE_SET_VERSION,
                "y_phase_sep": row["y_phase_sep"],
                "Phase separation": row["Phase separation"],
                "Phase Type": row["Phase Type"],
                "amine1_name_canonical": row["amine1_name_canonical"],
                "amine2_name_canonical": row["amine2_name_canonical"],
                "solvent_name_canonical": row["solvent_name_canonical"],
                "amine_set_key": row["amine_set_key"],
                "system_type": row["system_type"],
                "contains_organic_solvent": row["contains_organic_solvent"],
                "is_aqueous_only": row["is_aqueous_only"],
                "has_amine2": row["has_amine2"],
                "has_solvent": row["has_solvent"],
                "H2O wt%": row["H2O wt%"],
                "aq_regime_flag": int(x_h2o > 0),
                "dry_nonaq_flag": int(x_h2o <= 0),
                "salt_out_defined_flag": int(wth > 0 and wts > 0),
                **feature_values,
            }
        )

    feature_rows.extend(
        [
            {
                "feature_set_version": FEATURE_SET_VERSION,
                "column_name": name,
                "feature_block": "baseline_backbone",
                "feature_layer": layer,
                "trainable": 1,
                "legacy_reference_status": "legacy_direction_reimplemented",
                "source_files": "final_modeling_dataset.csv; sigma_profiles_wide_selected.csv",
                "source_detail": detail,
                "construction": construction,
                "display_policy": "implicit_in_code_exported_only_in_modeling_matrix",
                "aq_nonaq_handling": (
                    "AQ branch active; dry rows set to 0"
                    if name in {"Water_ent", "Sal_Out"}
                    else "same_formula_all_regimes"
                ),
            }
            for name, layer, construction, detail, _ in feature_specs
        ]
    )
    feature_rows.extend(
        [
            {
                "feature_set_version": FEATURE_SET_VERSION,
                "column_name": name,
                "feature_block": "meta_only",
                "feature_layer": "meta",
                "trainable": 0,
                "legacy_reference_status": "current_project_metadata",
                "source_files": "final_modeling_dataset.csv",
                "source_detail": name,
                "construction": "Retained metadata for audit, split, or subgroup evaluation",
                "display_policy": "kept_in_modeling_table",
                "aq_nonaq_handling": "not_applicable",
            }
            for name in [
                "row_id",
                "feature_set_version",
                "y_phase_sep",
                "Phase separation",
                "Phase Type",
                "amine1_name_canonical",
                "amine2_name_canonical",
                "solvent_name_canonical",
                "amine_set_key",
                "system_type",
                "contains_organic_solvent",
                "is_aqueous_only",
                "has_amine2",
                "has_solvent",
                "H2O wt%",
                "aq_regime_flag",
                "dry_nonaq_flag",
                "salt_out_defined_flag",
            ]
        ]
    )

    modeling_df = pd.DataFrame(modeling_rows)
    feature_df = pd.DataFrame(feature_rows)
    feature_df["missing_count"] = feature_df["column_name"].map(modeling_df.isna().sum().to_dict()).fillna(0).astype(int)
    feature_df["is_constant"] = feature_df["column_name"].map(
        lambda c: int(c in modeling_df.columns and modeling_df[c].dropna().nunique() <= 1)
    )
    feature_df["retained_as_trainable"] = ((feature_df["trainable"] == 1) & (feature_df["is_constant"] == 0)).astype(int)
    return modeling_df, feature_df


def write_reports(modeling_df: pd.DataFrame, feature_df: pd.DataFrame) -> None:
    trainable = feature_df.loc[feature_df["retained_as_trainable"] == 1, "column_name"].tolist()
    report_text = f"""# Legacy Realigned Backbone V1

## Scope

This report documents `{FEATURE_SET_VERSION}`, a legacy-direction backbone reimplementation under the current repository rules. It uses neutral-side features only and does not inherit legacy reacted routing, legacy split logic, or legacy dry handling.

## Feature Set Shape

- L0 features: `4`
- L1 features: `4`
- L2 features: `3`
- L3 features: `11`
- Total trainable backbone features: `{len(trainable)}`

## Referenced Legacy Ideas

- L0/L1/L2/L3 layered backbone direction
- approximately 22-dimensional backbone target
- sigma-profile segmentation plus integration instead of raw bin fusion
- separate amine-side and medium-side descriptors followed by mismatch features

## Explicitly Not Inherited

- legacy reacted default lookup via `_Reacted`
- legacy LOCO split logic
- legacy model conclusions
- legacy dry rows passing through aqueous salt-out formulas unchanged
- legacy dependence on wide explicit sample-table display columns

## Current Project Realignments

- neutral-side only baseline
- strict amine-object-holdout is handled in the evaluation script, not inherited from legacy
- AQ/NONAQ special handling is explicit for `Water_ent` and `Sal_Out`
- L1-L3 are constructed implicitly in code and exported only into the modeling matrix
- reacted routing metadata remains outside the baseline
"""
    (ROOT / "reports" / "baseline_realign_legacy_v1.md").write_text(report_text, encoding="utf-8")

    l3_text = f"""# L3 Segmentation Design

## Goal

L3 uses true sigma-profile segmentation plus integration, not weighted summary replacement.

## Sigma Regions

- donor region: `sigma <= {SIGMA_DONOR_THRESHOLD:.4f}`
- nonpolar region: `{SIGMA_DONOR_THRESHOLD:.4f} < sigma < {SIGMA_ACCEPTOR_THRESHOLD:.4f}`
- acceptor region: `sigma >= {SIGMA_ACCEPTOR_THRESHOLD:.4f}`

## Integration Method

- neutral object profiles are taken from `sigma_profiles_wide_selected.csv`
- each molecule uses its 60-bin area vector directly
- sample-level mixture profiles are built from `moles * area_bin`
- mixture profiles are normalized after summation
- width is represented by the second moment `sum(p_i * sigma_i^2)`

## Legacy-Aligned L3 Outputs

- `Am_acc`, `Am_don`, `Am_np`, `Am_wid`
- `Sv_acc`, `Sv_don`, `Sv_np`, `Sv_wid`
- `Delta_S_np = |Am_np - Sv_np|`
- `Delta_S_acc = |Am_acc - Sv_acc|`
- `Sal_Out = x_H2O * solvent_nonpolar_fraction` only when both water and solvent are present

## AQ / NONAQ Handling

- AQ rows: medium profile includes solvent plus water; `Water_ent` and `Sal_Out` are active when defined
- dry rows: medium profile excludes water automatically; `Water_ent = 0`; `Sal_Out = 0`
- aqueous-only rows with no organic solvent: `Sal_Out = 0` because solvent-driven salt-out is undefined
"""
    (ROOT / "reports" / "l3_segmentation_design.md").write_text(l3_text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    mapping = build_canonical_map()
    main_df = load_main_table(mapping)
    validate_reactive_dictionary(main_df)
    object_df, sigma_axis = build_neutral_object_table(mapping)
    object_lookup = object_table_to_lookup(object_df)
    modeling_df, feature_df = build_legacy_aligned_table(main_df, object_lookup, sigma_axis)
    modeling_df.to_csv(OUTPUTS_DIR / "baseline_modeling_table_legacy_v1.csv", index=False)
    feature_df.to_csv(OUTPUTS_DIR / "baseline_feature_dictionary_legacy_v1.csv", index=False)
    write_reports(modeling_df, feature_df)


if __name__ == "__main__":
    main()
