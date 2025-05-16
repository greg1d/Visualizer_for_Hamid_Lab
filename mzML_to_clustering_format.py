import os
import warnings

import numpy as np
import pandas as pd
import psutil
from numba import njit, prange, set_num_threads
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


def calculate_CCS_for_mzML_files(df, beta, tfix):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    df = df.rename(columns={"Drift Time (ms)": "DT", "m/z": "m/z_ion"})

    if "DT" not in df.columns or "m/z_ion" not in df.columns:
        raise ValueError("DataFrame must contain 'DT' and 'm/z_ion' columns.")

    DT_gas = 28.006148
    gamma = (df["m/z_ion"] / (DT_gas + df["m/z_ion"])) ** 0.5
    adjusted_dt = df["DT"] - tfix
    df["CCS (Å^2)"] = adjusted_dt / (beta * gamma)
    return df


@njit(parallel=True, fastmath=True)
def calculate_distance_block(
    mz_values,
    rt_values,
    ccs_values,
    ppm_tolerance,
    rt_tolerance,
    ccs_tolerance,
    eps_cutoff,
    max_pairs,
):
    n = len(mz_values)

    # Pre-allocated arrays for efficient parallelization
    row_indices = np.empty(max_pairs, dtype=np.int32)
    col_indices = np.empty(max_pairs, dtype=np.int32)
    dist_values = np.empty(max_pairs, dtype=np.float32)

    count = 0

    for i in prange(n):
        for j in range(i + 1, n):  # Only calculate upper triangle (symmetric matrix)
            mz_diff = abs(mz_values[i] - mz_values[j])
            rt_diff = abs(rt_values[i] - rt_values[j])
            ccs_diff = abs(ccs_values[i] - ccs_values[j])

            mz_dist = mz_diff / (mz_values[i] * ppm_tolerance)
            rt_dist = rt_diff / rt_tolerance
            ccs_dist = ccs_diff / (ccs_values[i] * ccs_tolerance)

            if mz_dist <= 1.1 and rt_dist <= 1.1 and ccs_dist <= 1.1:
                dist_value = np.sqrt(mz_dist**2 + rt_dist**2 + ccs_dist**2)
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
    df, ppm_tolerance, rt_tolerance, ccs_tolerance, eps_cutoff
):
    mz_values = df["m/z_ion"].values.astype(np.float32)
    rt_values = df["Retention Time (sec)"].values.astype(np.float32)
    ccs_values = df["CCS (Å^2)"].values.astype(np.float32)

    # Calculate max_pairs dynamically based on system memory
    max_pairs = calculate_max_pairs()

    row_indices, col_indices, dist_values = calculate_distance_block(
        mz_values,
        rt_values,
        ccs_values,
        ppm_tolerance,
        rt_tolerance,
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


def perform_optimized_clustering(df, sparse_matrix, eps_cutoff):
    if not isinstance(sparse_matrix, csr_matrix):
        sparse_matrix = sparse_matrix.tocsr()

    dbscan = DBSCAN(eps=eps_cutoff, min_samples=2, metric="precomputed", n_jobs=-1)
    labels = dbscan.fit_predict(sparse_matrix)

    # Retain Base Peak Intensity in the final DataFrame
    df["Cluster"] = labels
    return df


if __name__ == "__main__":
    # Example Data - Ensure this is a DataFrame
    df = {
        "Spectrum Number": [1, 2, 3],
        "Drift Time (ms)": [12.5, 12.5, 13.2],
        "Retention Time (sec)": [300, 332, 310],
        "m/z_ion": [1000, 1000.01, 520.3],
        "Base Peak Intensity": [1000, 1200, 1100],
        "CCS (Å^2)": [100, 100, 102],
    }

    df = pd.DataFrame(df)
    df_original = df.copy()  # Preserve the original DataFrame with Base Peak Intensity

    tfix = -0.067817
    beta = 0.138218

    eps_cutoff = 1.732050808  # Adjusted EPS cutoff value for three dimensions
    ppm_tolerance = 1e-5
    rt_tolerance = 30
    ccs_tolerance = 0.02

    sparse_matrix = create_distance_matrix_sparse(
        df, ppm_tolerance, rt_tolerance, ccs_tolerance, eps_cutoff
    )

    df = perform_optimized_clustering(df, sparse_matrix, eps_cutoff)
    # Restore Base Peak Intensity from the original DataFrame
    print(df)
