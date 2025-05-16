import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from numba import njit


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


@njit(fastmath=True)
def calculate_distance_block(
    mz_values,
    rt_values,
    ccs_values,
    ppm_tolerance,
    rt_tolerance,
    ccs_tolerance,
    eps_cutoff,
):
    """
    Vectorized distance calculation block without dynamic lists.
    """
    n = len(mz_values)
    row_indices = []
    col_indices = []
    dist_values = []

    for i in range(n):
        mz_diff = np.abs(mz_values[i] - mz_values)
        rt_diff = np.abs(rt_values[i] - rt_values)
        ccs_diff = np.abs(ccs_values[i] - ccs_values)

        mz_dist = mz_diff / (mz_values[i] * ppm_tolerance)
        rt_dist = rt_diff / rt_tolerance
        ccs_dist = ccs_diff / (ccs_values[i] * ccs_tolerance)
        valid_mask = (mz_dist <= 1.1) & (rt_dist <= 1.1) & (ccs_dist <= 1.1)
        # Only retain valid distances
        mz_dist = mz_dist[valid_mask]
        rt_dist = rt_dist[valid_mask]
        ccs_dist = ccs_dist[valid_mask]

        dist_row = np.sqrt(mz_dist**2 + rt_dist**2 + ccs_dist**2)
        below_cutoff = dist_row <= eps_cutoff

        row_indices.extend(np.full(np.sum(below_cutoff), i))
        col_indices.extend(np.where(below_cutoff)[0])
        dist_values.extend(dist_row[below_cutoff])

    return np.array(row_indices), np.array(col_indices), np.array(dist_values)


def create_distance_matrix_sparse(
    df, ppm_tolerance, rt_tolerance, ccs_tolerance, eps_cutoff
):
    """
    Create a sparse distance matrix using block-wise vectorized calculations.
    """
    mz_values = df["m/z_ion"].values.astype(np.float32)
    rt_values = df["Retention Time (sec)"].values.astype(np.float32)
    ccs_values = df["CCS (Å^2)"].values.astype(np.float32)

    # Use Numba-accelerated block distance calculation
    row_indices, col_indices, dist_values = calculate_distance_block(
        mz_values,
        rt_values,
        ccs_values,
        ppm_tolerance,
        rt_tolerance,
        ccs_tolerance,
        eps_cutoff,
    )

    # Create a sparse matrix using COOrdinate format (efficient for construction)
    sparse_matrix = coo_matrix(
        (dist_values, (row_indices, col_indices)),
        shape=(len(mz_values), len(mz_values)),
        dtype=np.float32,
    ).tocsr()

    print(
        f"Non-zero elements of the sparse distance matrix: {sparse_matrix.count_nonzero()}"
    )
    return sparse_matrix


if __name__ == "__main__":
    # Example Data - Ensure this is a DataFrame
    df = {
        "Spectrum Number": [1, 2, 3],
        "Drift Time (ms)": [12.5, 12.5, 13.2],
        "Retention Time (sec)": [300, 303, 310],
        "m/z_ion": [1000, 1000.009, 520.3],
        "Base Peak Intensity": [1000, 1200, 1100],
        "CCS (Å^2)": [100, 100.5, 102],
    }

    df = pd.DataFrame(df)
    tfix = -0.067817
    beta = 0.138218

    eps_cutoff = 1.732  # Adjusted EPS cutoff value for three dimensions
    ppm_tolerance = 1e-5
    rt_tolerance = 30
    ccs_tolerance = 0.02

    sparse_matrix = create_distance_matrix_sparse(
        df, ppm_tolerance, rt_tolerance, ccs_tolerance, eps_cutoff
    )
    print(sparse_matrix)
