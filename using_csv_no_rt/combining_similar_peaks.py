import numpy as np
import pandas as pd
from clustering_csv import (
    calculate_CCS_for_csv_files,
)
from reading_in_csv import read_csv_with_numeric_header


def calculate_CCS_for_csv_files2(df, beta, tfix):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    DT_gas = 28.006148
    gamma = (df["m/z"] / (DT_gas + df["m/z"])) ** 0.5
    adjusted_dt = df["DT"] - tfix
    df["CCS (Å^2)"] = adjusted_dt / (beta * gamma)
    return df


def combine_similar_clusters(
    df,
    beta,
    tfix,
    ppm_tolerance=1.5e-5,  # 15 ppm
    ccs_tolerance=0.02,  # 2%
):
    from scipy.spatial import KDTree

    if df.empty:
        return df

    df = df.reset_index(drop=True).copy()

    if df.empty:
        return df

    # Calculate CCS
    DT_gas = 28.006148
    gamma = (df["m/z"] / (DT_gas + df["m/z"])) ** 0.5
    adjusted_dt = df["DT (ms)"] - tfix
    df["CCS (Å^2)"] = adjusted_dt / (beta * gamma)

    used = set()
    combined_rows = []

    data = df[["m/z", "CCS (Å^2)"]].replace([np.inf, -np.inf], np.nan).dropna()
    valid_indices = data.index.to_list()
    df = df.loc[valid_indices].reset_index(drop=True)
    data = df[["m/z", "CCS (Å^2)"]].values
    tree = KDTree(data)

    for i in range(len(df)):
        if i in used:
            continue

        mz_i, ccs_i = data[i]

        idxs = tree.query_ball_point([mz_i, ccs_i], r=np.inf)
        group = [
            j
            for j in idxs
            if j not in used
            and abs(data[j][0] - mz_i) <= data[j][0] * ppm_tolerance
            and abs(data[j][1] - ccs_i) <= data[j][1] * ccs_tolerance
        ]

        used.update(group)
        group_df = df.iloc[group]

        avg_mz = group_df["m/z"].mean()
        avg_ccs = group_df["CCS (Å^2)"].mean()
        sum_intensity = group_df["Peak Intensity"].sum()
        avg_drift = group_df["DT (ms)"].mean()

        combined_rows.append(
            {
                "m/z": round(avg_mz, 4),
                "CCS (Å^2)": round(avg_ccs, 2),
                "Peak Intensity": sum_intensity,
                "Drift": round(avg_drift, 2),
            }
        )

    return pd.DataFrame(combined_rows)


def merge_drift_bins_by_mass(df, drift_step=0.12, drift_tol=0.01):
    if not {"Mass", "Drift", "Abundance"}.issubset(df.columns):
        raise ValueError("Missing required columns.")

    result_rows = []

    for mass_val, group in df.groupby("Mass"):
        group = group.sort_values("Drift").reset_index(drop=True)
        used = set()

        for i in range(len(group)):
            if i in used:
                continue

            ref_drift = group.loc[i, "Drift"]
            # Find continuous steps of 0.12 ± drift_tol
            sub_group = group[
                ((group["Drift"] - ref_drift) / drift_step)
                .round()
                .sub((group["Drift"] - ref_drift) / drift_step)
                .abs()
                < drift_tol
            ]

            sub_group_indices = sub_group.index
            used.update(sub_group_indices)

            total_abundance = sub_group["Abundance"].sum()
            drift_of_max = sub_group.loc[sub_group["Abundance"].idxmax(), "Drift"]

            result_rows.append(
                {
                    "Mass": round(mass_val, 5),
                    "Drift": round(drift_of_max, 2),
                    "Abundance": total_abundance,
                }
            )

    return pd.DataFrame(result_rows)


if __name__ == "__main__":
    df = read_csv_with_numeric_header("hamid_labs_data/SB_testing.csv")
    tfix = -0.067817
    beta = 0.138218
    df = calculate_CCS_for_csv_files(df, beta, tfix)

    cluster_totals = merge_drift_bins_by_mass(df)
    df = calculate_CCS_for_csv_files(cluster_totals, beta, tfix)

    print(df)
