import warnings

# Suppress specific pyOpenMS OPENMS_DATA_PATH warning
warnings.filterwarnings(
    "ignore",
    message=".*OPENMS_DATA_PATH environment variable already exists.*",
    category=UserWarning,
)


import pandas as pd  # noqa: E402
from pyopenms import *  # Must come after warning suppression  # noqa: E402, F403

from determining_peak_centroiding import (  # noqa: E402
    calculate_cluster_relative_intensity,
    extract_cluster_by_mz_dt,
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


def reassign_similar_clusters(
    cluster_df,
    full_df,
    beta,
    tfix,
    ppm_tolerance=1.5e-5,
    rt_tolerance=0.5,
    ccs_tolerance=0.02,
):
    """
    Reassigns similar clusters to a shared new cluster ID in full_df based on similarity in m/z, RT, and CCS.

    Parameters:
    - cluster_df (pd.DataFrame): Cluster-level summary. Must include columns for m/z, RT, and DT.
    - full_df (pd.DataFrame): Raw data with 'Cluster', 'Min m/z', 'Drift Time (ms)', and 'Base Peak Intensity'.
    - beta, tfix: Calibration constants for CCS calculation.
    - ppm_tolerance: m/z tolerance in parts-per-million.
    - rt_tolerance: RT tolerance in minutes.
    - ccs_tolerance: CCS tolerance in Å^2.

    Returns:
    - pd.DataFrame: A copy of full_df with updated Cluster assignments in 'Cluster', and original IDs in 'Original Cluster'.
    """
    import numpy as np
    from scipy.spatial import KDTree

    if cluster_df.empty:
        return full_df.copy()

    # Normalize column names if needed
    column_map = {
        "m/z_ion_center": "m/z",
        "Retention Time (sec)_center": "RT (min)",
        "DT_center": "DT (ms)",
    }
    cluster_df = cluster_df.rename(
        columns={k: v for k, v in column_map.items() if k in cluster_df.columns}
    )

    # Check required columns
    required = ["Cluster", "m/z", "RT (min)", "DT (ms)"]
    for col in required:
        if col not in cluster_df.columns:
            raise ValueError(f"[ERROR] Missing required column: '{col}' in cluster_df.")

    cluster_df = cluster_df.reset_index(drop=True).copy()
    full_df = full_df.copy()

    # Step 1: Estimate CCS for each cluster
    DT_gas = 28.006148
    gamma = (cluster_df["m/z"] / (DT_gas + cluster_df["m/z"])) ** 0.5
    adjusted_dt = cluster_df["DT (ms)"] - tfix
    cluster_df["CCS (Å^2)"] = adjusted_dt / (beta * gamma)

    # Step 2: Build KDTree for m/z, RT, CCS comparison
    used = set()
    cluster_to_new_id = {}
    new_cluster_id = 0

    data = cluster_df[["m/z", "RT (min)", "CCS (Å^2)"]].values
    tree = KDTree(data)

    for i in range(len(cluster_df)):
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
        group_ids = cluster_df.iloc[group]["Cluster"].unique()

        for cid in group_ids:
            cluster_to_new_id[cid] = new_cluster_id

        new_cluster_id += 1

    # Step 3: Apply new cluster ID map to full_df
    full_df["Original Cluster"] = full_df["Cluster"]
    full_df["Cluster"] = (
        full_df["Cluster"].map(cluster_to_new_id).fillna(full_df["Cluster"]).astype(int)
    )

    return full_df


if __name__ == "__main__":
    tfix = -0.067817
    beta = 0.138218
    exclude_noise_flag = True

    # Example Data - Ensure this is a DataFrame
    file_path = select_file()
    ppm_tolerance = 2e-5
    rt_tolerance = 30
    ccs_tolerance = 0.02

    # Load or process the data (caching with SQLite in .temp)
    full_df = load_or_process_data(
        file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
    )

    # Optionally exclude noise points
    full_df = exclude_noise_points(full_df, exclude_noise=exclude_noise_flag)

    # Extract cluster by specified criteria
    cluster_df = calculate_cluster_relative_intensity(full_df)
    cluster_df = generate_cluster_centroid_report(cluster_df)

    full_df_reassigned = reassign_similar_clusters(
        cluster_df,
        full_df,
        beta,
        tfix,
        ppm_tolerance,
        rt_tolerance,
        ccs_tolerance,
    )
    new_centroid_df = generate_cluster_centroid_report(full_df_reassigned)
    cluster_df = extract_cluster_by_mz_dt(
        new_centroid_df,
        mz_center=311.27,
        mz_tolerance=0.03,
        dt_center=34.5,
        dt_tolerance=1,
        mz_col="m/z_ion_center",
        dt_col="DT_center",
    )
