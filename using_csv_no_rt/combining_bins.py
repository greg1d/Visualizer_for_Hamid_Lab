import numpy as np
import pandas as pd
from Binning import process_mass_ccs_data_fixed_drift


def combine_bins_numpy(df):
    df["Abundance"] = pd.to_numeric(df["Abundance"], errors="coerce").fillna(0)
    df["Mass Bin"] = pd.to_numeric(df["Mass Bin"], errors="coerce")
    df["CCS Bin"] = pd.to_numeric(df["CCS Bin"], errors="coerce")

    mass_bins = df["Mass Bin"].to_numpy()
    ccs_bins = df["CCS Bin"].to_numpy()
    abundance = df["Abundance"].to_numpy()
    drift = df["Drift"].to_numpy()
    ms_mode = df["MS Mode"].to_numpy()

    keys = np.core.records.fromarrays([mass_bins, ccs_bins], names="mass_bin,ccs_bin")
    unique_keys, inverse_indices = np.unique(keys, return_inverse=True)

    summed_abundance = np.zeros(len(unique_keys))
    first_drift = np.zeros(len(unique_keys))
    first_ms_mode = np.zeros(len(unique_keys), dtype=ms_mode.dtype)

    for i, idx in enumerate(inverse_indices):
        summed_abundance[idx] += abundance[i]
        if first_drift[idx] == 0 and first_ms_mode[idx] == 0:
            first_drift[idx] = drift[i]
            first_ms_mode[idx] = ms_mode[i]

    combined_df = pd.DataFrame(
        {
            "Mass Bin": unique_keys.mass_bin,
            "CCS Bin": unique_keys.ccs_bin,
            "Abundance": summed_abundance,
            "Drift": first_drift,
            "MS Mode": first_ms_mode,
        }
    )

    return combined_df


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

    combined_df = combine_bins_numpy(df)
    combined_df.to_csv("xxx_sample_combined.csv", index=False)
    print("Saved combined output to xxx_sample_combined.csv")
