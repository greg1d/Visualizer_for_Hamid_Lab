import numpy as np
import pandas as pd
from reading_in_csv import calculate_CCS, read_and_label


def create_fixed_bins(min_value, max_value, ppm_tolerance):
    """
    Create reproducible mass bin edges using ppm-based widths.
    Returns bin edges (array) and labels (string ranges).
    """
    edges = [min_value]
    value = min_value
    while value < max_value:
        width = value * ppm_tolerance / 1_000_000
        value = min(value + width, max_value)
        edges.append(value)
    labels = [f"{edges[i]:.6f}–{edges[i + 1]:.6f}" for i in range(len(edges) - 1)]
    return np.array(edges), labels


def create_ccs_bins(min_ccs, max_ccs, ccs_tolerance):
    """
    Create CCS bins using percentage-based width.
    """
    edges = [min_ccs]
    ccs = min_ccs
    while ccs < max_ccs:
        width = max(ccs * ccs_tolerance, 1e-12)  # avoid zero width
        ccs = min(ccs + width, max_ccs)
        edges.append(ccs)
    labels = [f"{edges[i]:.6f}–{edges[i + 1]:.6f}" for i in range(len(edges) - 1)]
    return np.array(edges), labels


def assign_bins(df, column_name, bin_edges, bin_labels):
    """
    Vectorized assignment of bins using pd.cut (unordered to avoid ValueError).
    """
    df = df.copy()
    df[f"{column_name} Bin"] = pd.cut(
        df[column_name],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=False,
        ordered=False,  # unordered labels
    )
    df[f"{column_name} Coordinate"] = df[f"{column_name} Bin"]
    return df


def process_mass_ccs_data(
    ms1_file,
    ms2_file,
    output_location,
    beta,
    tfix,
    combine_dfs=True,
    min_mass=50,
    max_mass=1750,
    mass_tolerance_ppm=10,
    min_ccs=0,
    max_ccs_factor=60,
    ccs_tolerance=0.02,  # 2%
):
    # Read and combine CSVs
    df = read_and_label(ms1_file, ms2_file, combine_dfs=combine_dfs)

    # Calculate CCS
    df = calculate_CCS(df, beta, tfix)

    # --- Mass binning ---
    mass_edges, mass_labels = create_fixed_bins(min_mass, max_mass, mass_tolerance_ppm)
    df = assign_bins(df, "Mass", mass_edges, mass_labels)

    # --- CCS binning ---
    max_ccs = max_mass * max_ccs_factor
    ccs_edges, ccs_labels = create_ccs_bins(min_ccs, max_ccs, ccs_tolerance)
    df = assign_bins(df, "CCS", ccs_edges, ccs_labels)

    # Save output
    df.to_csv(output_location, index=False)
    print(f"Saved output to {output_location}")
    return df


if __name__ == "__main__":
    ms1_file = "using_csv_no_rt/Low_only.csv"
    ms2_file = "using_csv_no_rt/High_only.csv"
    output_location = "xxx_sample.csv"
    beta = 0.138218
    tfix = -0.067817

    df = process_mass_ccs_data(
        ms1_file=ms1_file,
        ms2_file=ms2_file,
        output_location=output_location,
        beta=beta,
        tfix=tfix,
        combine_dfs=True,
    )
