import numpy as np
import pandas as pd
from reading_in_csv import read_and_label


def create_fixed_bins(min_value, max_value, bin_width):
    edges = np.arange(min_value, max_value + bin_width, bin_width)
    centers = (edges[:-1] + edges[1:]) / 2  # central value for each bin
    return edges, centers


def assign_bins_with_centers(df, column_name, bin_edges, bin_centers):
    df = df.copy()
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    # Assign each value to a bin index
    bin_idx = np.digitize(df[column_name], bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_centers) - 1)  # avoid out-of-bounds
    # Store central bin value
    df[f"{column_name} Bin"] = bin_centers[bin_idx]
    return df


def assign_bins_and_ccs(
    df,
    beta,
    tfix,
    min_mass=50,
    max_mass=2000,
    mass_bin_width=0.01,
    min_drift=0,
    max_drift=60,
    drift_bin_width=0.04,
):
    # Mass bins
    mass_edges, mass_centers = create_fixed_bins(min_mass, max_mass, mass_bin_width)
    df = assign_bins_with_centers(df, "Mass", mass_edges, mass_centers)

    # Drift bins
    drift_edges, drift_centers = create_fixed_bins(
        min_drift, max_drift, drift_bin_width
    )
    df = assign_bins_with_centers(df, "Drift", drift_edges, drift_centers)

    # CCS calculation
    DT_gas = 28.006148
    gamma = np.sqrt(df["Mass Bin"] / (DT_gas + df["Mass Bin"]))
    df["CCS (Å^2)"] = (df["Drift Bin"] - tfix) / (beta * gamma)
    return df


def filter_by_abundance(df, min_abundance=1):
    df = df.copy()
    df["Abundance"] = pd.to_numeric(df["Abundance"], errors="coerce")
    return df[df["Abundance"] >= min_abundance]


def clean_final_df(df):
    df = df.copy()
    df["Mass"] = df["Mass"].round(4)
    df["Drift"] = df["Drift"].round(2)
    df["Mass Bin"] = df["Mass Bin"].round(4)
    df["Drift Bin"] = df["Drift Bin"].round(2)
    df["CCS (Å^2)"] = df["CCS (Å^2)"].round(2)
    return df[
        ["Mass", "Drift", "Abundance", "MS Mode", "Mass Bin", "Drift Bin", "CCS (Å^2)"]
    ]


def process_mass_ccs_data_fixed_drift(
    ms1_file,
    ms2_file,
    output_location,
    beta,
    tfix,
    combine_dfs=True,
    min_abundance=1,
    min_mass=50,
    max_mass=2000,
    mass_bin_width=0.01,
    min_drift=0,
    max_drift=60,
    drift_bin_width=0.04,
):
    df = read_and_label(ms1_file, ms2_file, combine_dfs=combine_dfs)
    df = filter_by_abundance(df, min_abundance)
    df = assign_bins_and_ccs(
        df,
        beta,
        tfix,
        min_mass,
        max_mass,
        mass_bin_width,
        min_drift,
        max_drift,
        drift_bin_width,
    )
    df = clean_final_df(df)
    return df


if __name__ == "__main__":
    ms1_file = "using_csv_no_rt/Low_only.csv"
    ms2_file = "using_csv_no_rt/High_only.csv"
    output_location = "test_sample.csv"
    beta = 0.138218
    tfix = -0.067817

    min_mass = 50
    max_mass = 2000
    mass_bin_width = 0.01

    min_drift = 0
    max_drift = 60
    drift_bin_width = 0.04

    min_abundance = 20

    df = process_mass_ccs_data_fixed_drift(
        ms1_file=ms1_file,
        ms2_file=ms2_file,
        output_location=output_location,
        beta=beta,
        tfix=tfix,
        combine_dfs=True,
        min_abundance=min_abundance,
        min_mass=min_mass,
        max_mass=max_mass,
        mass_bin_width=mass_bin_width,
        min_drift=min_drift,
        max_drift=max_drift,
        drift_bin_width=drift_bin_width,
    )
