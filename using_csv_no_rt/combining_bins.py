import numpy as np
import pandas as pd
from Binning import process_mass_ccs_data_fixed_drift


def process_and_combine(
    ms1_file,
    ms2_file=None,
    output_location="output.csv",
    beta=0.138218,
    tfix=0,
    min_mass=50,
    max_mass=2000,
    mass_bin_width=0.01,
    min_drift=0,
    max_drift=60,
    drift_bin_width=0.04,
    min_abundance=20,
    combine_dfs=False,  # <-- new flag
):
    # Step 1: Process the raw MS data into bins
    df = process_mass_ccs_data_fixed_drift(
        ms1_file=ms1_file,
        ms2_file=ms2_file,
        output_location=output_location,
        beta=beta,
        tfix=tfix,
        combine_dfs=combine_dfs,  # <-- pass through directly
        min_abundance=min_abundance,
        min_mass=min_mass,
        max_mass=max_mass,
        mass_bin_width=mass_bin_width,
        min_drift=min_drift,
        max_drift=max_drift,
        drift_bin_width=drift_bin_width,
    )

    # Step 2: Combine identical bins (Mass Bin + Drift Bin + MS Mode)
    grouped = df.groupby(["Mass Bin", "Drift Bin", "MS Mode"], as_index=False)
    combined = grouped.apply(
        lambda g: pd.Series(
            {
                "Mass": np.average(g["Mass"], weights=g["Abundance"]),
                "Drift": np.average(g["Drift"], weights=g["Abundance"]),
                "Abundance": g["Abundance"].sum(),
                "CCS (Å^2)": np.average(g["CCS (Å^2)"], weights=g["Abundance"]),
            }
        )
    )

    # Keep column order consistent
    combined = combined[
        ["Mass", "Drift", "Abundance", "MS Mode", "Mass Bin", "Drift Bin", "CCS (Å^2)"]
    ]

    # Round numeric columns
    combined["Mass"] = combined["Mass"].round(4)
    combined["Drift"] = combined["Drift"].round(2)
    combined["Mass Bin"] = combined["Mass Bin"].round(4)
    combined["Drift Bin"] = combined["Drift Bin"].round(2)
    combined["CCS (Å^2)"] = combined["CCS (Å^2)"].round(2)

    # Optionally convert Abundance to integer
    combined["Abundance"] = combined["Abundance"].astype(int)
    combined.to_csv(output_location, index=False)
    return combined


if __name__ == "__main__":
    # Example with combine_dfs = True
    df_final = process_and_combine(
        ms1_file="using_csv_no_rt/Low_only.csv",
        ms2_file="using_csv_no_rt/High_only.csv",
        output_location="test_sample.csv",
        beta=0.138218,
        tfix=-0.067817,
        min_mass=50,
        max_mass=2000,
        mass_bin_width=0.01,
        min_drift=0,
        max_drift=60,
        drift_bin_width=0.04,
        min_abundance=1000,
        combine_dfs=False,  # <-- explicit choice
    )
    print(df_final)
