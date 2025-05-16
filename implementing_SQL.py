import hashlib
import os
import sqlite3
import time

import pandas as pd

from mzML_to_clustering_format import (
    calculate_CCS_for_mzML_files,
    create_distance_matrix_sparse,
    perform_optimized_clustering,
)
from Open_MS_config import extract_spectrum_data, handle_uploaded_file, select_file


def sum_base_peak_intensity_per_cluster(df):
    """
    Sum the Base Peak Intensity for each cluster.
    """
    if "Cluster" not in df.columns:
        raise ValueError("DataFrame must contain 'Cluster' column.")

    cluster_sums = df.groupby("Cluster")["Base Peak Intensity"].sum().reset_index()
    cluster_sums = cluster_sums.rename(
        columns={"Base Peak Intensity": "Total Base Peak Intensity"}
    )

    df = pd.merge(df, cluster_sums, on="Cluster", how="left")
    return df


def ensure_temp_directory():
    """
    Ensure the .temp directory exists and perform cache cleanup.
    """
    temp_dir = ".temp"
    os.makedirs(temp_dir, exist_ok=True)
    cleanup_old_cache(temp_dir)
    return temp_dir


def cleanup_old_cache(temp_dir, max_cache_age=7 * 24 * 60 * 60):  # 7 days
    """
    Clean up old cached files in the .temp directory.
    """
    current_time = time.time()
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_cache_age:
                print(f"Deleting old cache file: {file_path}")
                os.remove(file_path)


def generate_cache_key(
    file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
):
    """
    Generate a unique cache key based on file path and instrument parameters.
    """
    param_string = (
        f"{file_path}_{tfix}_{beta}_{ppm_tolerance}_{rt_tolerance}_{ccs_tolerance}"
    )
    return hashlib.md5(param_string.encode()).hexdigest()


def load_or_process_data(
    file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
):
    """
    Load processed data from SQL or process if not already cached.
    """
    # Ensure the .temp directory exists
    temp_dir = ensure_temp_directory()
    db_path = os.path.join(temp_dir, "processed_data.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_data (
            file_hash TEXT PRIMARY KEY,
            data BLOB
        )
    """)

    # Generate a cache key based on file path and parameters
    cache_key = generate_cache_key(
        file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
    )

    # Check if the processed data is already cached
    cursor.execute("SELECT data FROM processed_data WHERE file_hash = ?", (cache_key,))
    result = cursor.fetchone()

    if result:
        print("Loading processed data from cache...")
        df_pickle_path = result[0]
        try:
            df = pd.read_pickle(df_pickle_path)
            print("Cached data loaded successfully.")
        except (FileNotFoundError, Exception) as e:
            print(f"Cache file not found or corrupted. Reprocessing... {e}")
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
    """
    Reprocess data and save it to the cache.
    """
    try:
        exp = handle_uploaded_file(file_path)
        df = extract_spectrum_data(exp)
        df = pd.DataFrame(df)

        # Apply CCS Calculation using user-specified tfix and beta
        df = calculate_CCS_for_mzML_files(df, beta, tfix)

        sparse_matrix = create_distance_matrix_sparse(
            df, ppm_tolerance, rt_tolerance, ccs_tolerance
        )
        df = perform_optimized_clustering(df, sparse_matrix)

        # Sum Base Peak Intensity for each cluster
        df = sum_base_peak_intensity_per_cluster(df)

        # Cache the processed DataFrame in the .temp directory
        df_pickle_path = os.path.join(temp_dir, f"df_cache_{cache_key}.pkl")
        df.to_pickle(df_pickle_path)
        cursor.execute(
            "INSERT OR REPLACE INTO processed_data (file_hash, data) VALUES (?, ?)",
            (cache_key, df_pickle_path),
        )
        conn.commit()
        print(f"Processed data cached to {df_pickle_path}.")
    except Exception as e:
        print(f"Error during data processing: {e}")
        raise

    return df


def exclude_noise_points(df, exclude_noise=True):
    """
    Exclude noise points (Cluster -1) from the DataFrame.
    """
    if exclude_noise:
        df = df[df["Cluster"] != -1].reset_index(drop=True)

    return df


def pick_a_cluster_for_debugging(df):
    """
    Extract and display the 50th largest cluster for debugging.
    """
    if not df.empty:
        cluster_sizes = df["Cluster"].value_counts().sort_values(ascending=False)

        if len(cluster_sizes) >= 50:
            target_cluster = cluster_sizes.index[49]  # 50th largest cluster (0-indexed)
            cluster_df = df[df["Cluster"] == target_cluster]
            return cluster_df
    return pd.DataFrame()


if __name__ == "__main__":
    # User-Editable Parameters
    tfix = -0.067817
    beta = 0.138218
    exclude_noise_flag = True  # Set this to True to exclude noise points (Cluster -1)

    # Example Data - Ensure this is a DataFrame
    file_path = select_file()
    ppm_tolerance = 1e-5
    rt_tolerance = 30
    ccs_tolerance = 0.01

    # Load or process the data (caching with SQLite in .temp)
    df = load_or_process_data(
        file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
    )

    # Optionally exclude noise points
    df = exclude_noise_points(df, exclude_noise=exclude_noise_flag)

    # Debugging: Extract the 50th largest cluster for analysis
    df = pick_a_cluster_for_debugging(df)
    print("\nCluster for Debugging:")
    print(df)
