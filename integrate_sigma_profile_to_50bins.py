#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
把原始 sigma profile 长表积分压缩到 50 个 bins，并与 neutral xTB 主特征表合并。

重要说明：
1. 输入必须是原始 sigma profile 长表，且至少包含 `name`, `sigma`, `profile` 三列。
2. 如果没有原始 sigma profile 长表，这一步不能执行。
3. 本脚本只新增输出文件，不修改任何现有原始数据文件。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="积分压缩 sigma profile 到 50 bins，并合并到 neutral 主特征表。")
    parser.add_argument("--sigma-long", required=True, help="原始 sigma profile 长表路径。")
    parser.add_argument(
        "--feature-master",
        default="neutral_xtb_rebuild/feature_exports/neutral_feature_master_from_xtb.csv",
        help="neutral 主特征表路径。",
    )
    parser.add_argument(
        "--sigma50-out",
        default="neutral_xtb_rebuild/feature_exports/sigma_profile_50bins.csv",
        help="50 bins 输出路径。",
    )
    parser.add_argument(
        "--merged-out",
        default="neutral_xtb_rebuild/feature_exports/neutral_feature_master_with_sigma50.csv",
        help="与 neutral 主特征表合并后的输出路径。",
    )
    parser.add_argument("--bins", type=int, default=50, help="目标 bin 数，默认 50。")
    parser.add_argument("--sigma-min", type=float, default=-0.025, help="积分区间下界，默认 -0.025。")
    parser.add_argument("--sigma-max", type=float, default=0.025, help="积分区间上界，默认 0.025。")
    return parser


def validate_sigma_long(df: pd.DataFrame) -> None:
    required_cols = {"name", "sigma", "profile"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        missing_str = ", ".join(sorted(missing_cols))
        raise ValueError(
            f"原始 sigma profile 长表缺少必要列: {missing_str}。"
            "这一步必须基于至少包含 name、sigma、profile 三列的原始长表执行。"
        )


def integrate_profile_to_bins(sigma_values: np.ndarray, profile_values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    用积分而不是简单取点来压缩单个分子的 sigma profile。

    对每个 bin：
    1. 取 bin 左右边界与 bin 内原始 sigma 采样点组成局部网格。
    2. 用线性插值得到这些网格上的 profile 值。
    3. 用梯形积分计算区间面积。
    4. 再除以 bin 宽度，得到该 bin 的平均 profile 强度。
    """
    order = np.argsort(sigma_values)
    sigma_sorted = np.asarray(sigma_values[order], dtype=float)
    profile_sorted = np.asarray(profile_values[order], dtype=float)

    finite_mask = np.isfinite(sigma_sorted) & np.isfinite(profile_sorted)
    sigma_sorted = sigma_sorted[finite_mask]
    profile_sorted = profile_sorted[finite_mask]
    if sigma_sorted.size == 0:
        return np.full(len(bin_edges) - 1, np.nan, dtype=float)

    if sigma_sorted.size == 1:
        out = np.zeros(len(bin_edges) - 1, dtype=float)
        sigma0 = sigma_sorted[0]
        value0 = profile_sorted[0]
        for i in range(len(bin_edges) - 1):
            left, right = bin_edges[i], bin_edges[i + 1]
            if left <= sigma0 <= right:
                out[i] = float(value0)
        return out

    dedup = pd.DataFrame({"sigma": sigma_sorted, "profile": profile_sorted}).groupby("sigma", as_index=False)["profile"].mean()
    sigma_sorted = dedup["sigma"].to_numpy(dtype=float)
    profile_sorted = dedup["profile"].to_numpy(dtype=float)

    out = np.zeros(len(bin_edges) - 1, dtype=float)
    for i in range(len(bin_edges) - 1):
        left = float(bin_edges[i])
        right = float(bin_edges[i + 1])
        inside_mask = (sigma_sorted > left) & (sigma_sorted < right)
        grid = np.concatenate(([left], sigma_sorted[inside_mask], [right]))
        grid = np.unique(grid)
        interp_profile = np.interp(grid, sigma_sorted, profile_sorted, left=profile_sorted[0], right=profile_sorted[-1])
        area = np.trapz(interp_profile, grid)
        out[i] = float(area / (right - left))
    return out


def compress_sigma_profile(sigma_long_df: pd.DataFrame, bins: int, sigma_min: float, sigma_max: float) -> pd.DataFrame:
    if bins <= 0:
        raise ValueError("bins 必须是正整数。")
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max 必须大于 sigma_min。")

    working_df = sigma_long_df.copy()
    working_df["name"] = working_df["name"].astype(str)
    working_df["sigma"] = pd.to_numeric(working_df["sigma"], errors="coerce")
    working_df["profile"] = pd.to_numeric(working_df["profile"], errors="coerce")
    working_df = working_df.dropna(subset=["name", "sigma", "profile"])

    bin_edges = np.linspace(sigma_min, sigma_max, bins + 1)
    bin_columns = [f"sigma_bin_{i:02d}" for i in range(bins)]

    rows: List[dict] = []
    for name, group in working_df.groupby("name", sort=True):
        integrated = integrate_profile_to_bins(
            sigma_values=group["sigma"].to_numpy(dtype=float),
            profile_values=group["profile"].to_numpy(dtype=float),
            bin_edges=bin_edges,
        )
        row = {"name": name}
        row.update({col: float(val) for col, val in zip(bin_columns, integrated)})
        rows.append(row)

    return pd.DataFrame(rows)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_input_path(path_str: str) -> Path:
    """
    将输入路径解析为可用路径。

    规则：
    1. 绝对路径直接使用。
    2. 相对路径优先相对于当前工作目录解释。
    3. 如果当前工作目录下不存在，再尝试相对于项目根目录解释。
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return PROJECT_ROOT / path


def resolve_output_path(path_str: str) -> Path:
    """
    将输出路径解析为可写路径。

    输出相对路径默认相对于项目根目录展开，这样从项目根目录执行脚本时更直观，
    从其他目录执行时也不容易把结果写到意外位置。
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = build_parser().parse_args()

    sigma_long_path = resolve_input_path(args.sigma_long)
    feature_master_path = resolve_input_path(args.feature_master)
    sigma50_out_path = resolve_output_path(args.sigma50_out)
    merged_out_path = resolve_output_path(args.merged_out)

    if not sigma_long_path.exists():
        raise FileNotFoundError(
            "没有找到原始 sigma profile 长表，因此这一步不能执行。"
            f"请先提供包含 name、sigma、profile 三列的输入文件: {sigma_long_path}"
        )

    sigma_long_df = pd.read_csv(sigma_long_path)
    validate_sigma_long(sigma_long_df)
    sigma50_df = compress_sigma_profile(
        sigma_long_df=sigma_long_df,
        bins=args.bins,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )

    ensure_parent_dir(sigma50_out_path)
    sigma50_df.to_csv(sigma50_out_path, index=False, encoding="utf-8-sig")

    if not feature_master_path.exists():
        raise FileNotFoundError(
            "没有找到 neutral_feature_master_from_xtb.csv，无法执行合并。"
            f"请先生成主表: {feature_master_path}"
        )

    feature_master_df = pd.read_csv(feature_master_path)
    if "name" not in feature_master_df.columns:
        raise ValueError("neutral_feature_master_from_xtb.csv 缺少 name 列，无法按分子名称合并。")

    merged_df = feature_master_df.merge(sigma50_df, on="name", how="left")
    ensure_parent_dir(merged_out_path)
    merged_df.to_csv(merged_out_path, index=False, encoding="utf-8-sig")

    print(f"已输出 50 bins sigma profile: {sigma50_out_path}")
    print(f"已输出合并主表: {merged_out_path}")


if __name__ == "__main__":
    main()
