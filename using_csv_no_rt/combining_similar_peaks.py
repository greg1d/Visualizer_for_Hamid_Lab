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


def collapse_peak_features_dt_dimension(df):
    """
    Collapse features by Mass and Drift proximity (± drift_window).
    For each group, sum Abundance and return the row with max Abundance as representative.

    Parameters:
    - df: DataFrame with ['Mass', 'Drift', 'Abundance']
    - drift_window: max drift distance within group (default 0.5)

    Returns:
    - DataFrame with one row per collapsed group
    """
    if not {"Mass", "Drift", "Abundance"}.issubset(df.columns):
        raise ValueError("Missing required columns.")
    drift_window = 0.5
    df = df.sort_values(["Mass", "Drift"]).reset_index(drop=True)
    collapsed = []
    used = set()

    for mass_val, group in df.groupby("Mass"):
        group = group.sort_values("Drift").reset_index()
        for i, row in group.iterrows():
            if row["index"] in used:
                continue

            center_drift = row["Drift"]
            # Select neighbors within ± drift_window
            mask = (group["Drift"] >= center_drift - drift_window) & (
                group["Drift"] <= center_drift + drift_window
            )
            neighbors = group[mask]
            neighbor_indices = neighbors["index"].tolist()
            used.update(neighbor_indices)

            total_abundance = neighbors["Abundance"].sum()
            rep_row = neighbors.loc[neighbors["Abundance"].idxmax()]
            collapsed.append(
                {
                    "Mass": rep_row["Mass"],
                    "Drift": rep_row["Drift"],
                    "Abundance": total_abundance,
                }
            )

    return pd.DataFrame(collapsed)


def identify_peak_features_mass_dimension(df, ppm_tolerance=1e-5):
    """
    Identify peaks in the mass dimension using a shell tolerance for neighbors.

    A point is considered a peak if its abundance is greater than the mean + N*std
    of neighboring points that are within ± drift_window AND whose mass lies between
    mass * ppm_tolerance * 3 and mass * ppm_tolerance * 5 from the center point.

    Parameters:
    - df: DataFrame with columns ['Mass', 'Drift', 'Abundance']
    - ppm_tolerance: base ppm tolerance for shell window
    - drift_window: ± window in Drift units to consider neighbors
    - threshold_multiplier: number of standard deviations above mean to call a peak

    Returns:
    - DataFrame with identified peak features
    """
    if not {"Mass", "Drift", "Abundance"}.issubset(df.columns):
        raise ValueError("Missing required columns.")
    drift_window = 0.24
    threshold_multiplier = 3
    df = df.sort_values(["Mass", "Drift"]).reset_index(drop=True)
    peak_rows = []

    for idx in range(len(df)):
        center_mass = df.loc[idx, "Mass"]
        center_drift = df.loc[idx, "Drift"]
        center_abundance = df.loc[idx, "Abundance"]

        min_mass_diff = center_mass * ppm_tolerance * 3
        max_mass_diff = center_mass * ppm_tolerance * 5

        mask = (
            (df["Drift"] >= center_drift - drift_window)
            & (df["Drift"] <= center_drift + drift_window)
            & (df.index != idx)
            & (df["Mass"].sub(center_mass).abs() >= min_mass_diff)
            & (df["Mass"].sub(center_mass).abs() <= max_mass_diff)
        )

        neighborhood = df[mask]
        if len(neighborhood) < 2:
            continue

        mean = neighborhood["Abundance"].mean()
        std = neighborhood["Abundance"].std()

        if center_abundance > (mean + threshold_multiplier * std):
            peak_rows.append(df.loc[idx])

    return pd.DataFrame(peak_rows)


def condense_mass_drift_features(features_df, ppm_tolerance=1e-5):
    """
    For a given set of features, group those within 10 ppm mass and ±drift_window
    and keep only the row with the highest abundance in each group.

    Parameters:
    - features_df: DataFrame with peak features ['Mass', 'Drift', 'Abundance', ...]
    - ppm_tolerance: mass tolerance as ppm
    - drift_window: drift grouping tolerance (e.g., 0.24)

    Returns:
    - Condensed DataFrame of maximal abundance representatives
    """
    import numpy as np
    from scipy.spatial import KDTree

    drift_window = 0.24
    if features_df.empty:
        return features_df

    features_df = features_df.copy().reset_index(drop=True)
    used = set()
    condensed_rows = []

    data = features_df[["Mass", "Drift"]].values
    tree = KDTree(data)

    for i in range(len(features_df)):
        if i in used:
            continue

        center_mass = features_df.loc[i, "Mass"]
        center_drift = features_df.loc[i, "Drift"]
        mass_tol = center_mass * ppm_tolerance

        idxs = tree.query_ball_point([center_mass, center_drift], r=np.inf)
        group = [
            j
            for j in idxs
            if j not in used
            and abs(features_df.loc[j, "Mass"] - center_mass) <= mass_tol * 3
            and abs(features_df.loc[j, "Drift"] - center_drift) <= drift_window
        ]

        used.update(group)
        group_df = features_df.loc[group]
        max_idx = group_df["Abundance"].idxmax()
        condensed_rows.append(features_df.loc[max_idx])

    return pd.DataFrame(condensed_rows)


def run_feature_identifying_workflow(df, tfix, beta, ppm_tolerance):
    cluster_totals = collapse_peak_features_dt_dimension(df)

    cluster_totals = identify_peak_features_mass_dimension(
        cluster_totals, ppm_tolerance
    )
    cluster_totals = condense_mass_drift_features(cluster_totals, ppm_tolerance)
    cluster_totals = calculate_CCS_for_csv_files(cluster_totals, beta, tfix)

    return cluster_totals


if __name__ == "__main__":
    df = read_csv_with_numeric_header("using_csv_no_rt/testing_data.csv")
    tfix = -0.067817
    beta = 0.138218
    ppm_tolerance = 1e-5

    cluster_totals = run_feature_identifying_workflow(df, tfix, beta, ppm_tolerance)
    print(cluster_totals)
