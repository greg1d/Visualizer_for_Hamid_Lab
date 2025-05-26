import warnings

# Suppress specific pyOpenMS OPENMS_DATA_PATH warning
warnings.filterwarnings(
    "ignore",
    message=".*OPENMS_DATA_PATH environment variable already exists.*",
    category=UserWarning,
)


import numpy as np
import pandas as pd  # noqa: E402
from pyopenms import *  # Must come after warning suppression  # noqa: E402, F403

from determining_peak_centroiding import (  # noqa: E402
    calculate_cluster_relative_intensity,
    generate_cluster_centroid_report,
)
from implementing_SQL import (  # noqa: E402
    exclude_noise_points,
    load_or_process_data,
)
from Open_MS_config import select_file  # noqa: E402


def make_clusters_pretty(df):
    """
    Formats the cluster DataFrame and computes CCS values.

    Parameters:
    - df (pd.DataFrame): Cluster-level DataFrame.
    - beta (float): CCS calibration constant.
    - tfix (float): CCS calibration offset.

    Returns:
    - pd.DataFrame: Formatted and CCS-calculated DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    required_cols = ["m/z_ion_center", "Retention Time (sec)_center", "DT_center"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    df = df.copy()

    # Step 1: Rename and transform
    df["m/z"] = df["m/z_ion_center"].round(4)
    df["RT (min)"] = (df["Retention Time (sec)_center"] / 60).round(2)
    df["DT (ms)"] = df["DT_center"]

    # Drop originals
    df.drop(
        columns=["m/z_ion_center", "Retention Time (sec)_center", "DT_center"],
        inplace=True,
    )

    # Step 2: Rename for
    df.rename(columns={"DT (ms)": "DT", "m/z": "Min m/z"}, inplace=True)

    # Step 5: Drop unwanted columns
    cols_to_drop = ["Amplitude_mz", "Amplitude_dt", "Amplitude_rt"]
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    # Step 6: Rename back for readability
    df.rename(columns={"DT": "DT (ms)", "Min m/z": "m/z"}, inplace=True)

    # Step 7: Reorder columns
    desired_order = [
        "Cluster",
        "RT (min)",
        "m/z",
        "DT (ms)",
        "Base Peak Intensity",
        "MS Level",
        "Sigma_rt",
    ]
    existing_cols = [col for col in desired_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_cols]
    df = df[existing_cols + remaining_cols]  # ordered + remaining

    return df


def crude_filter_clusters(
    df,
    mass_min=0,
    mass_max=np.inf,
    rt_min=0,
    rt_max=np.inf,
    dt_min=0,
    dt_max=np.inf,
    intensity_min=0,
):
    """
    Apply crude bounds to filter out rows outside of specified ranges.
    Assumes 'm/z', 'RT (min)', 'DT (ms)', and 'Peak Intensity' columns exist.

    Returns:
    - Filtered DataFrame.
    """
    return df[
        (df["m/z"] >= mass_min)
        & (df["m/z"] <= mass_max)
        & (df["RT (min)"] >= rt_min)
        & (df["RT (min)"] <= rt_max)
        & (df["DT (ms)"] >= dt_min)
        & (df["DT (ms)"] <= dt_max)
        & (df["Peak Intensity"] >= intensity_min)
    ].copy()


def combine_similar_clusters(
    df,
    beta,
    tfix,
    ppm_tolerance=1.5e-5,
    rt_tolerance=0.5,
    ccs_tolerance=0.02,
    mass_min=0,
    mass_max=np.inf,
    rt_min=0,
    rt_max=np.inf,
    dt_min=0,
    dt_max=np.inf,
    intensity_min=0,
):
    """
    Combines rows in the DataFrame that fall within specified tolerances
    for m/z (ppm), RT (min), and CCS (Å²), and recalculates CCS.
    """
    import numpy as np
    from scipy.spatial import KDTree

    if df.empty:
        return df

    df = df.reset_index(drop=True).copy()

    # Rename raw columns
    rename_map = {
        "m/z_ion_center": "m/z",
        "Retention Time (sec)_center": "RT (min)",
        "DT_center": "DT (ms)",
    }
    df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True
    )

    # Convert RT from seconds to minutes
    df["RT (min)"] = df["RT (min)"] / 60

    # Apply crude filters
    df = crude_filter_clusters(
        df,
        mass_min=mass_min,
        mass_max=mass_max,
        rt_min=rt_min,
        rt_max=rt_max,
        dt_min=dt_min,
        dt_max=dt_max,
        intensity_min=intensity_min,
    )

    if df.empty:
        return df

    # Estimate CCS
    DT_gas = 28.006148
    gamma = (df["m/z"] / (DT_gas + df["m/z"])) ** 0.5
    adjusted_dt = df["DT (ms)"] - tfix
    df["CCS (Å^2)"] = adjusted_dt / (beta * gamma)

    used = set()
    combined_rows = []

    data = (
        df[["m/z", "RT (min)", "CCS (Å^2)"]].replace([np.inf, -np.inf], np.nan).dropna()
    )
    valid_indices = data.index.to_list()
    df = df.loc[valid_indices].reset_index(drop=True)
    data = df[["m/z", "RT (min)", "CCS (Å^2)"]].values
    tree = KDTree(data)

    for i in range(len(df)):
        if i in used:
            continue

        mz_i, rt_i, ccs_i = data[i]
        mz_tol = mz_i * ppm_tolerance

        idxs = tree.query_ball_point([mz_i, rt_i, ccs_i], r=np.inf)
        group = [
            j
            for j in idxs
            if j not in used
            and abs(data[j][0] - mz_i) <= mz_tol
            and abs(data[j][1] - rt_i) <= rt_tolerance
            and abs(data[j][2] - ccs_i) <= ccs_tolerance
        ]

        used.update(group)
        group_df = df.iloc[group]

        avg_mz = group_df["m/z"].mean()
        avg_rt = group_df["RT (min)"].mean()
        avg_dt = group_df["DT (ms)"].mean()
        sum_intensity = group_df["Peak Intensity"].sum()
        avg_ms_level = (
            int(round(group_df["MS Level"].mean()))
            if "MS Level" in group_df.columns and not group_df["MS Level"].isna().all()
            else -1
        )
        avg_sigma_rt = (
            group_df["Sigma_rt"].mean()
            if "Sigma_rt" in group_df.columns and not group_df["Sigma_rt"].isna().all()
            else np.nan
        )

        gamma_avg = (avg_mz / (DT_gas + avg_mz)) ** 0.5
        adj_dt = avg_dt - tfix
        new_ccs = adj_dt / (beta * gamma_avg)

        combined_rows.append(
            {
                "RT (min)": round(avg_rt, 2),
                "m/z": round(avg_mz, 4),
                "DT (ms)": round(avg_dt, 2),
                "CCS (Å^2)": round(new_ccs, 2),
                "Peak Intensity": sum_intensity,
                "MS Level": avg_ms_level,
                "Sigma_rt": avg_sigma_rt,
            }
        )

    return pd.DataFrame(combined_rows)


if __name__ == "__main__":
    tfix = -0.067817
    beta = 0.138218
    exclude_noise_flag = True

    # Example Data - Ensure this is a DataFrame
    file_path = select_file()
    ppm_tolerance = 1.5e-5
    rt_tolerance = 30
    ccs_tolerance = 0.02

    # Load or process the data (caching with SQLite in .temp)
    df = load_or_process_data(
        file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
    )

    # Optionally exclude noise points
    cluster_df = exclude_noise_points(df, exclude_noise=exclude_noise_flag)

    # Extract cluster by specified criteria
    cluster_df = calculate_cluster_relative_intensity(cluster_df)
    cluster_df = generate_cluster_centroid_report(cluster_df)
    mass_min = 200
    mass_max = 1000
    rt_min = 0.5
    rt_max = 10
    dt_min = 0
    dt_max = 60
    intensity_min = 0

    cluster_df = combine_similar_clusters(
        cluster_df,
        beta,
        tfix,
        ppm_tolerance=1.5e-5,
        rt_tolerance=0.5,
        ccs_tolerance=0.02,
        mass_min=mass_min,
        mass_max=mass_max,
        rt_min=rt_min,
        rt_max=rt_max,
        dt_min=dt_min,
        dt_max=dt_max,
        intensity_min=intensity_min,
    )
    cluster_df.to_csv("combined_clusters.csv", index=False)
