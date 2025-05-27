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


def identify_peak_features(df):
    """
    Identify peaks where a point's Abundance is greater than the mean + N*std
    of all points within ± drift_window (excluding itself).

    Parameters:
    - df: DataFrame with columns ['Mass', 'Drift', 'Abundance']
    - drift_window: the ± window (in Drift units) to consider neighbors
    - threshold_multiplier: multiplier for standard deviation to define a peak

    Returns:
    - DataFrame with identified peak features
    """
    if not {"Mass", "Drift", "Abundance"}.issubset(df.columns):
        raise ValueError("Missing required columns.")
    drift_window = 0.5
    threshold_multiplier = 3
    df = df.sort_values(["Mass", "Drift"]).reset_index(drop=True)
    peak_rows = []

    for mass_val, group in df.groupby("Mass"):
        group = group.sort_values("Drift").reset_index(drop=True)

        for i in range(len(group)):
            center_drift = group.loc[i, "Drift"]
            center_abundance = group.loc[i, "Abundance"]

            # Define neighbor range (excluding self)
            mask = (
                (group["Drift"] >= center_drift - drift_window)
                & (group["Drift"] <= center_drift + drift_window)
                & (group["Drift"] != center_drift)
            )

            neighborhood = group[mask]
            if len(neighborhood) < 2:
                continue  # skip if no sufficient context

            mean = neighborhood["Abundance"].mean()
            std = neighborhood["Abundance"].std()

            if center_abundance > (mean + threshold_multiplier * std):
                peak_rows.append(group.loc[i])

    return pd.DataFrame(peak_rows)


if __name__ == "__main__":
    df = read_csv_with_numeric_header("using_csv_no_rt/testing_data.csv")
    tfix = -0.067817
    beta = 0.138218
    df = calculate_CCS_for_csv_files(df, beta, tfix)

    cluster_totals = identify_peak_features(df)

    print(cluster_totals)
