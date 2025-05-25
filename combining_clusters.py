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
from mzML_to_clustering_format import calculate_CCS_for_mzML_files  # noqa: E402
from Open_MS_config import select_file  # noqa: E402


def make_clusters_pretty(df, beta, tfix):
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

    # Step 2: Rename for CCS
    df.rename(columns={"DT (ms)": "DT", "m/z": "Min m/z"}, inplace=True)

    # Step 3: Compute CCS
    df = calculate_CCS_for_mzML_files(df, beta, tfix)

    # Step 4: Round DT and CCS
    df["DT"] = df["DT"].round(2)
    df["CCS (Å^2)"] = df["CCS (Å^2)"].round(2)

    # Step 5: Drop unwanted columns
    cols_to_drop = ["Amplitude_mz", "Amplitude_dt", "Amplitude_rt", "Cluster"]
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    # Step 6: Rename back for readability
    df.rename(columns={"DT": "DT (ms)", "Min m/z": "m/z"}, inplace=True)

    # Step 7: Reorder columns
    desired_order = [
        "RT (min)",
        "m/z",
        "DT (ms)",
        "CCS (Å^2)",
        "Base Peak Intensity",
        "MS Level",
        "Sigma_rt",
    ]
    existing_cols = [col for col in desired_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_cols]
    df = df[existing_cols + remaining_cols]  # ordered + remaining

    return df


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
    cluster_df = extract_cluster_by_mz_dt(
        cluster_df,
        mz_center=311.27,
        mz_tolerance=0.03,
        dt_center=34.5,
        dt_tolerance=1,
        mz_col="m/z_ion_center",
        dt_col="DT_center",
    )

    cluster_df = make_clusters_pretty(cluster_df, beta, tfix)
    print(cluster_df)
