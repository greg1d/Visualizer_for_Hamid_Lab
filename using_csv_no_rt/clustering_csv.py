import os
import warnings

import numpy as np
import pandas as pd
import psutil
from numba import njit, prange, set_num_threads
from reading_in_csv import read_csv_with_numeric_header
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cluster import DBSCAN

# Suppress all warnings (Numba, Sklearn, and others)
warnings.filterwarnings("ignore")

# Set maximum CPU utilization
num_threads = os.cpu_count()
set_num_threads(num_threads)


# Dynamically determine max_pairs based on available system memory
def calculate_max_pairs():
    total_memory = psutil.virtual_memory().total  # Total system memory in bytes
    safe_memory = total_memory * 0.7  # Use only 70% of the total memory
    pair_size_bytes = np.dtype(np.int32).itemsize * 2 + np.dtype(np.float32).itemsize
    max_pairs = int(safe_memory // pair_size_bytes)
    return max_pairs


def calculate_CCS_for_csv_files(df, beta, tfix):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    df = df.rename(columns={"Mass": "Min m/z", "Drift": "DT"})

    if "DT" not in df.columns or "Min m/z" not in df.columns:
        raise ValueError("DataFrame must contain 'DT' and 'Min m/z' columns.")

    DT_gas = 28.006148
    gamma = (df["Min m/z"] / (DT_gas + df["Min m/z"])) ** 0.5
    adjusted_dt = df["DT"] - tfix
    df["CCS (Å^2)"] = adjusted_dt / (beta * gamma)
    return df


@njit(parallel=True, fastmath=True)
def calculate_distance_block(
    mz_values,
    ccs_values,
    ppm_tolerance,
    ccs_tolerance,
    eps_cutoff,
    max_pairs,
):
    n = len(mz_values)
    eps_cutoff = 3 ** (0.5)
    # Pre-allocated arrays for efficient parallelization
    row_indices = np.empty(max_pairs, dtype=np.int32)
    col_indices = np.empty(max_pairs, dtype=np.int32)
    dist_values = np.empty(max_pairs, dtype=np.float32)

    count = 0

    for i in prange(n):
        for j in range(i + 1, n):  # Only calculate upper triangle (symmetric matrix)
            mz_diff = abs(mz_values[i] - mz_values[j])
            ccs_diff = abs(ccs_values[i] - ccs_values[j])

            mz_dist = mz_diff / (mz_values[i] * ppm_tolerance)
            ccs_dist = ccs_diff / (ccs_values[i] * ccs_tolerance)

            if mz_dist <= 1.1 and ccs_dist <= 1.1:
                dist_value = np.sqrt(mz_dist**2 + ccs_dist**2)
                if dist_value <= eps_cutoff:
                    if count >= max_pairs:  # Exceeding calculated limit
                        raise MemoryError("Exceeded dynamically calculated max_pairs.")

                    row_indices[count] = i
                    col_indices[count] = j
                    dist_values[count] = dist_value
                    count += 1

    # Return only the filled part of the arrays
    return row_indices[:count], col_indices[:count], dist_values[:count]


def create_distance_matrix_sparse(
    df,
    ppm_tolerance,
    ccs_tolerance,
):
    mz_values = df["Min m/z"].values.astype(np.float32)
    ccs_values = df["CCS (Å^2)"].values.astype(np.float32)
    eps_cutoff = 2 ** (0.5)

    # Calculate max_pairs dynamically based on system memory
    max_pairs = calculate_max_pairs()

    row_indices, col_indices, dist_values = calculate_distance_block(
        mz_values,
        ccs_values,
        ppm_tolerance,
        ccs_tolerance,
        eps_cutoff,
        max_pairs=max_pairs,
    )

    # Create a sparse matrix using COOrdinate format (efficient for construction)
    sparse_matrix = coo_matrix(
        (dist_values, (row_indices, col_indices)),
        shape=(len(mz_values), len(mz_values)),
        dtype=np.float32,
    ).tocsr()

    return sparse_matrix


def perform_optimized_clustering(df, sparse_matrix):
    if not isinstance(sparse_matrix, csr_matrix):
        sparse_matrix = sparse_matrix.tocsr()
    eps_cutoff = 3 ** (0.5)

    dbscan = DBSCAN(eps=eps_cutoff, min_samples=2, metric="precomputed", n_jobs=-1)
    labels = dbscan.fit_predict(sparse_matrix)

    # Retain Base Peak Intensity in the final DataFrame
    df["Cluster"] = labels
    return df


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
    # Example Data - Ensure this is a DataFrame

    df = read_csv_with_numeric_header("hamid_labs_data/SB.csv")
    print(df.head())
    tfix = -0.067817
    beta = 0.138218
    df = calculate_CCS_for_csv_files(df, beta, tfix)

    ppm_tolerance = 1e-5
    ccs_tolerance = 0.02

    sparse_matrix = create_distance_matrix_sparse(df, ppm_tolerance, ccs_tolerance)
    df = perform_optimized_clustering(df, sparse_matrix)
