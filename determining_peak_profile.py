import hashlib
import os
import sqlite3

import pandas as pd

from mzML_to_clustering_format import (
    calculate_CCS_for_mzML_files,
    create_distance_matrix_sparse,
    perform_optimized_clustering,
)
from Open_MS_config import extract_spectrum_data, handle_uploaded_file, select_file


def sum_base_peak_intensity_per_cluster(df):
    if "Cluster" not in df.columns:
        raise ValueError("DataFrame must contain 'Cluster' column.")

    cluster_sums = df.groupby("Cluster")["Base Peak Intensity"].sum().reset_index()
    cluster_sums = cluster_sums.rename(
        columns={"Base Peak Intensity": "Total Base Peak Intensity"}
    )

    df = pd.merge(df, cluster_sums, on="Cluster", how="left")
    return df


def ensure_temp_directory():
    temp_dir = ".temp"
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def generate_cache_key(
    file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
):
    param_string = (
        f"{file_path}_{tfix}_{beta}_{ppm_tolerance}_{rt_tolerance}_{ccs_tolerance}"
    )
    return hashlib.md5(param_string.encode()).hexdigest()


def load_or_process_data(
    file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
):
    temp_dir = ensure_temp_directory()
    db_path = os.path.join(temp_dir, "processed_data.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_data (
            file_hash TEXT PRIMARY KEY,
            data BLOB
        )
    """)

    cache_key = generate_cache_key(
        file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
    )
    cursor.execute("SELECT data FROM processed_data WHERE file_hash = ?", (cache_key,))
    result = cursor.fetchone()

    if result:
        print("Loading processed data from cache...")
        df_pickle_path = result[0]
        try:
            df = pd.read_pickle(df_pickle_path)
            print("Cached data loaded successfully.")
        except (FileNotFoundError, Exception):
            print("Cache file not found or corrupted. Reprocessing...")
            df = reprocess_and_cache(
                file_path,
                tfix,
                beta,
                ppm_tolerance,
                rt_tolerance,
                ccs_tolerance,
                temp_dir,
                cache_key,
                cursor,
                conn,
            )
    else:
        print("Processing data...")
        df = reprocess_and_cache(
            file_path,
            tfix,
            beta,
            ppm_tolerance,
            rt_tolerance,
            ccs_tolerance,
            temp_dir,
            cache_key,
            cursor,
            conn,
        )

    conn.close()
    return df


def reprocess_and_cache(
    file_path,
    tfix,
    beta,
    ppm_tolerance,
    rt_tolerance,
    ccs_tolerance,
    temp_dir,
    cache_key,
    cursor,
    conn,
):
    exp = handle_uploaded_file(file_path)
    df = extract_spectrum_data(exp)
    df = pd.DataFrame(df)

    df = calculate_CCS_for_mzML_files(df, beta, tfix)

    sparse_matrix = create_distance_matrix_sparse(
        df, ppm_tolerance, rt_tolerance, ccs_tolerance
    )
    df = perform_optimized_clustering(df, sparse_matrix)
    df = sum_base_peak_intensity_per_cluster(df)

    df_pickle_path = os.path.join(temp_dir, f"df_cache_{cache_key}.pkl")
    df.to_pickle(df_pickle_path)
    cursor.execute(
        "INSERT OR REPLACE INTO processed_data (file_hash, data) VALUES (?, ?)",
        (cache_key, df_pickle_path),
    )
    conn.commit()
    print(f"Processed data cached to {df_pickle_path}.")
    return df


def exclude_noise_points(df, exclude_noise=True):
    if exclude_noise:
        df = df[df["Cluster"] != -1].reset_index(drop=True)
    return df


def extract_cluster_by_criteria(
    df,
    mz_target,
    dt_target,
    rt_target,
    mz_tolerance=0.1,
    dt_tolerance=1,
    rt_tolerance=50,
):
    """
    Extract a cluster that matches the specified m/z, DT, and RT criteria.
    """
    cluster_candidates = df[
        (df["m/z_ion"].between(mz_target - mz_tolerance, mz_target + mz_tolerance))
        & (df["DT"].between(dt_target - dt_tolerance, dt_target + dt_tolerance))
        & (
            df["Retention Time (sec)"].between(
                rt_target - rt_tolerance, rt_target + rt_tolerance
            )
        )
    ]

    if cluster_candidates.empty:
        print("No clusters found matching the criteria.")
        return pd.DataFrame()

    # Find the most representative cluster among the candidates
    selected_cluster = cluster_candidates["Cluster"].mode().iloc[0]
    cluster_df = df[df["Cluster"] == selected_cluster]
    print(f"Extracted Cluster: {selected_cluster} with {len(cluster_df)} rows.")
    return cluster_df


def extract_cluster_by_number(df, cluster_id):
    """
    Extract and return the DataFrame for a specific cluster number.
    """
    cluster_df = df[df["Cluster"] == cluster_id].copy()
    if cluster_df.empty:
        print(f"[INFO] Cluster {cluster_id} not found.")
    else:
        print(f"[INFO] Extracted Cluster {cluster_id} with {len(cluster_df)} rows.")
    return cluster_df


if __name__ == "__main__":
    # User-Editable Parameters
    tfix = -0.067817
    beta = 0.138218
    exclude_noise_flag = True

    # Example Data - Ensure this is a DataFrame
    file_path = select_file()
    ppm_tolerance = 1e-5
    rt_tolerance = 30
    ccs_tolerance = 0.02

    # Load or process the data (caching with SQLite in .temp)
    df = load_or_process_data(
        file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
    )

    # Optionally exclude noise points
    df = exclude_noise_points(df, exclude_noise=exclude_noise_flag)

    # Extract cluster by specified criteria
    cluster_df = extract_cluster_by_number(df, cluster_id=27142)

    print("\nExtracted Cluster by Criteria:")
    print(cluster_df)
