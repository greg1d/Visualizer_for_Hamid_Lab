import pandas as pd
from Open_MS_config import select_file, handle_uploaded_file, extract_spectrum_data
import numpy as np


def calculate_CCS_for_mzML_files(df, beta, tfix):
    """
    Calculate CCS for mzML files using the provided DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the spectrum data.
    beta (float): Beta value for CCS calculation - taken from IMBrowser (open Tune File in IMBrowser -> Integrate across whole RT range -> View -> CCS Calibration (Single Field) -> Select all mass peaks -> Find Drift Times -> Copy over mass values).
    tfix (float): Copy steps from getting the beta.

    Returns:
    pd.DataFrame: DataFrame with calculated CCS values.
    """
    # Ensure the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Rename columns for clarity
    df = df.rename(columns={"Drift Time (ms)": "DT", "m/z": "m/z_ion"})

    if "DT" not in df.columns or "m/z_ion" not in df.columns:
        raise ValueError("DataFrame must contain 'DT' and 'm/z_ion' columns.")

    # Constant for gas constant
    DT_gas = 28.006148

    # Calculate CCS using the EXACT specified formula
    gamma = (df["m/z_ion"] / (DT_gas + df["m/z_ion"])) ** 0.5
    adjusted_dt = df["DT"] - tfix
    df["CCS (Å^2)"] = adjusted_dt / (beta * gamma)
    return df


from scipy.sparse import coo_matrix


def create_condensed_distance_matrix(
    df,
    ppm_tolerance=1e-5,
    rt_tolerance=0.5,
    ccs_tolerance=0.02,
    eps_cutoff=3,
):
    """
    Create a sparse distance matrix for large datasets using block-wise calculations.

    Parameters:
    df (pd.DataFrame): DataFrame containing m/z, RT, and CCS values.
    ppm_tolerance (float): Tolerance for m/z differences (ppm).
    rt_tolerance (float): Tolerance for RT differences (seconds).
    ccs_tolerance (float): Tolerance for CCS differences.
    eps_cutoff (float): Distance cutoff for sparse storage.
    block_size (int): Number of rows to process in each block for memory efficiency.

    Returns:
    csr_matrix: Sparse distance matrix (CSR format).
    """
    
    mz_values = np.array(mz_values)
    rt_values = np.array(rt_values)
    ccs_values = np.array(ccs_values)

    n = len(mz_values)
    row_indices = []
    col_indices = []
    dist_values = []

    # Process in blocks for memory efficiency
    for i in range(0, n, block_size):
        end = min(i + block_size, n)

        # Current block values
        mz_block = mz_values[i:end]
        rt_block = rt_values[i:end]
        ccs_block = ccs_values[i:end]

        # Broadcast differences within the block to all other values
        for j in range(n):
            mz_diff = np.abs(mz_block - mz_values[j])
            rt_diff = np.abs(rt_block - rt_values[j])
            ccs_diff = np.abs(ccs_block - ccs_values[j])

            mz_dist = mz_diff / (mz_values[i] * ppm_tolerance)
            rt_dist = rt_diff / rt_tolerance
            ccs_dist = ccs_diff / (ccs_values[i] * ccs_tolerance)

            dist_row = np.sqrt(mz_dist**2 + rt_dist**2 + ccs_dist**2)

            # Apply eps_cutoff
            mask = dist_row <= eps_cutoff
            rows_in_block = np.arange(i, end)[mask]
            row_indices.extend(rows_in_block)
            col_indices.extend([j] * len(rows_in_block))
            dist_values.extend(dist_row[mask])

    # Create a sparse matrix using COOrdinate format (efficient for construction)
    sparse_matrix = coo_matrix(
        (dist_values, (row_indices, col_indices)), shape=(n, n), dtype=np.float32
    ).tocsr()

    print(
        f"Sparse matrix created with {sparse_matrix.count_nonzero()} non-zero elements."
    )
    return sparse_matrix


def perform_clustering(df, ppm_tolerance, rt_tolerance, ccs_tolerance):
    eps_cutoff = 1.732  # Adjusted EPS cutoff value for three dimensions
    mz_values = df["m/z_ion"].values
    rt_values = df["Retention Time (sec)"].values
    ccs_values = df["CCS (Å^2)"].values
    dist_matrix_sparse = create_condensed_distance_matrix(
        ppm_tolerance=ppm_tolerance,
        rt_tolerance=rt_tolerance,
        ccs_tolerance=ccs_tolerance,
        eps_cutoff=eps_cutoff,
    )
    return dist_matrix_sparse


if __name__ == "__main__":
    # Example Data - Ensure this is a DataFrame
    file_path = select_file()
    exp = handle_uploaded_file(file_path)
    df = extract_spectrum_data(exp)
    df = pd.DataFrame(df)
    tfix = -0.067817
    beta = 0.138218

    df = calculate_CCS_for_mzML_files(df, beta, tfix)

    eps_cutoff = 1.732  # Adjusted EPS cutoff value for three dimensions
    ppm_tolerance = 1e-5
    rt_tolerance = 0.5
    ccs_tolerance = 0.02

    df = create_condensed_distance_matrix(
        df, ppm_tolerance, rt_tolerance, ccs_tolerance
    )
    print(df)
