import numpy as np
import pandas as pd
from reading_in_csv import read_and_label


def create_fixed_bins(min_value, max_value, bin_width):
    edges = np.arange(min_value, max_value + bin_width, bin_width)
    labels = [f"{edges[i]:.6f}–{edges[i + 1]:.6f}" for i in range(len(edges) - 1)]
    return edges, labels


def assign_bins(df, column_name, bin_edges, bin_labels):
    df = df.copy()
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    df[f"{column_name} Bin"] = pd.cut(
        df[column_name],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=False,
        ordered=False,
    )
    df[f"{column_name} Coordinate"] = df[f"{column_name} Bin"]
    return df


def create_drift_bins(min_drift, max_drift, bin_width=0.04):
    edges = np.arange(min_drift, max_drift + bin_width, bin_width)
    labels = [f"{edges[i]:.4f}–{edges[i + 1]:.4f}" for i in range(len(edges) - 1)]
    return edges, labels


def assign_mass_bin_centers(df):
    def bin_center(cat):
        if pd.isna(cat):
            return np.nan
        start, end = str(cat).split("–")
        return (float(start) + float(end)) / 2

    df = df.copy()
    df["Mass Bin Center"] = df["Mass Bin"].apply(bin_center)
    df["Mass Bin Center"] = pd.to_numeric(df["Mass Bin Center"], errors="coerce")
    return df


def assign_drift_bins_and_ccs(
    df, beta, tfix, min_drift=0, max_drift=60, bin_width=0.04
):
    drift_edges, drift_labels = create_drift_bins(min_drift, max_drift, bin_width)
    df = assign_bins(df, "Drift", drift_edges, drift_labels)

    # Calculate numeric drift bin centers
    def drift_center(cat):
        if pd.isna(cat):
            return np.nan
        start, end = str(cat).split("–")
        return (float(start) + float(end)) / 2

    df["Drift Bin Center"] = df["Drift Bin"].apply(drift_center)
    df["Drift Bin Center"] = pd.to_numeric(df["Drift Bin Center"], errors="coerce")
    df = assign_mass_bin_centers(df)
    return df


def calculate_ccs_from_bins(df, beta, tfix):
    DT_gas = 28.006148
    gamma = (df["Mass Bin Center"] / (DT_gas + df["Mass Bin Center"])) ** 0.5
    adjusted_dt = df["Drift Bin Center"] - tfix
    df["CCS (Å^2)"] = adjusted_dt / (beta * gamma)
    return df


def filter_by_abundance(df, min_abundance=1):
    df = df.copy()
    df["Abundance"] = pd.to_numeric(df["Abundance"], errors="coerce")
    df = df[df["Abundance"] >= min_abundance]
    return df


def clean_final_df(df):
    df = df.copy()
    final_columns = [
        "Mass",
        "Drift",
        "Abundance",
        "MS Mode",
        "Mass Bin",
        "Drift Bin",
        "CCS (Å^2)",
    ]
    df = df[[col for col in final_columns if col in df.columns]]

    if "Mass" in df.columns:
        df["Mass"] = df["Mass"].round(4)
    if "Drift" in df.columns:
        df["Drift"] = df["Drift"].round(2)
    if "Mass Bin" in df.columns:
        df["Mass Bin"] = df["Mass Bin"].round(4)
    if "Drift Bin" in df.columns:
        df["Drift Bin"] = df["Drift Bin"].round(2)
    if "CCS (Å^2)" in df.columns:
        df["CCS (Å^2)"] = df["CCS (Å^2)"].round(2)

    return df


def process_mass_ccs_data_fixed_drift(
    ms1_file,
    ms2_file,
    output_location,
    beta,
    tfix,
    combine_dfs=True,
    min_abundance=1,
    min_mass=50,
    max_mass=1750,
    mass_bin_width=0.01,
    min_drift=0,
    max_drift=60,
    drift_bin_width=0.04,
):
    df = read_and_label(ms1_file, ms2_file, combine_dfs=combine_dfs)
    df = filter_by_abundance(df, min_abundance=min_abundance)

    # Mass binning
    mass_edges, mass_labels = create_fixed_bins(min_mass, max_mass, mass_bin_width)
    df = assign_bins(df, "Mass", mass_edges, mass_labels)

    # Drift binning and CCS
    df = assign_drift_bins_and_ccs(
        df, beta, tfix, min_drift, max_drift, drift_bin_width
    )
    df = calculate_ccs_from_bins(df, beta, tfix)

    # Clean and round final DataFrame
    df = clean_final_df(df)
    df.to_csv(output_location, index=False)
    print(f"Saved output to {output_location}")
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
