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
    """
    Sum the Base Peak Intensity for each cluster.

    Parameters:
    df (pd.DataFrame): DataFrame containing cluster information and Base Peak Intensity.

    Returns:
    pd.DataFrame: DataFrame with clusters and their total Base Peak Intensity.
    """
    if "Cluster" not in df.columns:
        raise ValueError("DataFrame must contain 'Cluster' column.")

    # Group by cluster and sum the Base Peak Intensity
    cluster_sums = df.groupby("Cluster")["Base Peak Intensity"].sum().reset_index()
    cluster_sums = cluster_sums.rename(
        columns={"Base Peak Intensity": "Total Base Peak Intensity"}
    )

    # Merge total intensity back to the main DataFrame
    df = pd.merge(df, cluster_sums, on="Cluster", how="left")
    return df


def load_or_process_data(
    file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
):
    """
    Load processed data from SQL or process if not already cached.
    """
    # Ensure the .temp directory exists
    temp_dir = ".temp"
    os.makedirs(temp_dir, exist_ok=True)

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

    # Generate a hash of the file path to uniquely identify it
    file_hash = hashlib.md5(file_path.encode()).hexdigest()

    # Check if the processed data is already cached
    cursor.execute("SELECT data FROM processed_data WHERE file_hash = ?", (file_hash,))
    result = cursor.fetchone()

    if result:
        print("Loading processed data from cache...")
        df_pickle_path = result[0]
        df = pd.read_pickle(df_pickle_path)
    else:
        print("Processing data...")
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
        df_pickle_path = os.path.join(temp_dir, f"df_cache_{file_hash}.pkl")
        df.to_pickle(df_pickle_path)
        cursor.execute(
            "INSERT INTO processed_data (file_hash, data) VALUES (?, ?)",
            (file_hash, df_pickle_path),
        )
        conn.commit()

    conn.close()
    return df


def exclude_noise_points(df, exclude_noise=True):
    """
    Exclude noise points (Cluster -1) from the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing cluster information.
    exclude_noise (bool): Whether to exclude noise points (Cluster -1).

    Returns:
    pd.DataFrame: DataFrame with or without noise points.
    """
    if exclude_noise:
        df = df[df["Cluster"] != -1].reset_index(drop=True)

    return df


def pick_a_cluster_for_debugging(df):
    """
    Extract and display the 50th largest cluster for debugging.

    Parameters:
    df (pd.DataFrame): DataFrame with cluster information.

    Returns:
    None
    """
    if not df.empty:
        cluster_sizes = df["Cluster"].value_counts().sort_values(ascending=False)

        if len(cluster_sizes) >= 5000:
            target_cluster = cluster_sizes.index[
                4999
            ]  # 50th largest cluster (0-indexed)

            cluster_df = df[df["Cluster"] == target_cluster]

    return cluster_df


if __name__ == "__main__":
    # User-Editable Parameters
    tfix = -0.067817
    beta = 0.138218
    exclude_noise_flag = True  # Set this to True to exclude noise points (Cluster -1)

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

    # Debugging: Extract the largest cluster for analysis
    df = pick_a_cluster_for_debugging(df)
    print(df)
