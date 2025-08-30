import numpy as np
import pandas as pd
from reading_in_csv import read_and_label


def create_fixed_bins(min_value, max_value, bin_width):
    """
    Create reproducible mass bin edges using a fixed width (user-specified).
    Returns bin edges (array) and labels (string ranges).
    """
    edges = np.arange(min_value, max_value + bin_width, bin_width)
    labels = [f"{edges[i]:.6f}–{edges[i + 1]:.6f}" for i in range(len(edges) - 1)]
    return edges, labels


def assign_bins(df, column_name, bin_edges, bin_labels):
    """
    Vectorized assignment of bins using pd.cut (unordered to avoid ValueError).
    Converts the column to numeric first.
    """
    df = df.copy()
    # Convert column to numeric
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
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


def create_drift_bins(min_drift, max_drift, bin_width=0.04):
    """
    Create drift bins with a fixed width (e.g., 0.04 ms).
    Returns bin edges and labels.
    """
    edges = np.arange(min_drift, max_drift + bin_width, bin_width)
    labels = [f"{edges[i]:.4f}–{edges[i + 1]:.4f}" for i in range(len(edges) - 1)]
    return edges, labels


def assign_drift_bins_and_ccs(
    df, beta, tfix, min_drift=0, max_drift=60, bin_width=0.04
):
    """
    Assign drift bins and calculate CCS from drift bin centers.
    """
    # Create drift bins and assign
    drift_edges, drift_labels = create_drift_bins(min_drift, max_drift, bin_width)
    df = assign_bins(df, "Drift", drift_edges, drift_labels)

    df = df.copy()

    # Convert Categorical to string and then to numeric bin centers
    def bin_center(cat):
        if pd.isna(cat):
            return np.nan
        s = str(cat)
        start, end = s.split("–")
        return (float(start) + float(end)) / 2

    df["Drift Bin Center"] = df["Drift Bin"].apply(bin_center)

    # Ensure numeric dtype
    df["Drift Bin Center"] = pd.to_numeric(df["Drift Bin Center"], errors="coerce")

    # Calculate CCS safely (NaNs will propagate)
    df["CCS"] = 1e-4 * (df["Drift Bin Center"] + tfix) ** beta

    return df


def filter_by_abundance(df, min_abundance=1):
    """
    Remove rows where Abundance (Intensity) is below min_abundance.
    Ensures Abundance column is numeric.
    """
    df = df.copy()
    df["Abundance"] = pd.to_numeric(df["Abundance"], errors="coerce")
    df = df[df["Abundance"] >= min_abundance]
    return df


def process_mass_ccs_data_fixed_drift(
    ms1_file,
    ms2_file,
    output_location,
    beta,
    tfix,
    combine_dfs=True,
    min_mass=50,
    max_mass=1750,
    mass_bin_width=0.01,  # user-specified single value
    min_drift=0,
    max_drift=60,
    drift_bin_width=0.04,
):
    # Read and combine CSVs
    df = read_and_label(ms1_file, ms2_file, combine_dfs=combine_dfs)
    df = filter_by_abundance(df, min_abundance=min_abundance)

    # --- Mass binning ---
    mass_edges, mass_labels = create_fixed_bins(min_mass, max_mass, mass_bin_width)
    df = assign_bins(df, "Mass", mass_edges, mass_labels)

    # --- Drift binning and CCS calculation ---
    df = assign_drift_bins_and_ccs(
        df, beta, tfix, min_drift, max_drift, drift_bin_width
    )

    # Save output
    df.to_csv(output_location, index=False)
    print(f"Saved output to {output_location}")
    return df


if __name__ == "__main__":
    # --- User-defined parameters ---
    ms1_file = "using_csv_no_rt/Low_only.csv"
    ms2_file = "using_csv_no_rt/High_only.csv"
    output_location = "test_sample.csv"
    beta = 0.138218
    tfix = -0.067817

    # Mass parameters
    min_mass = 50
    max_mass = 2000
    mass_bin_width = 0.01  # user-specified mass bin width

    # Drift/CCS parameters
    min_drift = 0
    max_drift = 60
    drift_bin_width = 0.04  # fixed width for drift binning

    min_abundance = 20

    df = process_mass_ccs_data_fixed_drift(
        ms1_file=ms1_file,
        ms2_file=ms2_file,
        output_location=output_location,
        beta=beta,
        tfix=tfix,
        combine_dfs=True,
        min_mass=min_mass,
        max_mass=max_mass,
        mass_bin_width=mass_bin_width,
        min_drift=min_drift,
        max_drift=max_drift,
        drift_bin_width=drift_bin_width,
    )
