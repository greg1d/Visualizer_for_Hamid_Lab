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


def create_distance_matrix_sparse(
    df,
    ppm_tolerance,
    rt_tolerance,
    ccs_tolerance,
    eps_cutoff,  # Only store distances below this cutoff
):
    """
    Create a sparse distance matrix for large datasets using vectorized calculations.
    """
    mz_values = df["m/z_ion"].values
    rt_values = df["Retention Time (sec)"].values
    ccs_values = df["CCS (Å^2)"].values
    mz_values = np.array(mz_values)
    rt_values = np.array(rt_values)
    ccs_values = np.array(ccs_values)

    n = len(mz_values)
    row_indices = []
    col_indices = []
    dist_values = []

    # Use vectorized calculations to compute all pairwise differences
    for i in range(n):
        mz_diff = np.abs(mz_values[i] - mz_values)
        rt_diff = np.abs(rt_values[i] - rt_values)
        ccs_diff = np.abs(ccs_values[i] - ccs_values)

        mz_dist = mz_diff / (mz_values[i] * ppm_tolerance)
        rt_dist = rt_diff / rt_tolerance
        ccs_dist = ccs_diff / (ccs_values[i] * ccs_tolerance)

        dist_row = np.sqrt(mz_dist**2 + rt_dist**2 + ccs_dist**2)

        # Apply the cutoff directly using a boolean mask
        if eps_cutoff is not None:
            below_cutoff = dist_row <= eps_cutoff
            non_zero_indices = np.where(below_cutoff)[0]
            row_indices.extend([i] * len(non_zero_indices))
            col_indices.extend(non_zero_indices)
            dist_values.extend(dist_row[non_zero_indices])
        else:
            row_indices.extend([i] * n)
            col_indices.extend(range(n))
            dist_values.extend(dist_row)

    # Create a sparse matrix using COOrdinate format (efficient for construction)
    sparse_matrix = coo_matrix(
        (dist_values, (row_indices, col_indices)), shape=(n, n), dtype=np.float32
    ).tocsr()

    print(
        f"Non-zero elements of the sparse distance matrix: {sparse_matrix.count_nonzero()}"
    )
    return sparse_matrix


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

    df = create_distance_matrix_sparse(
        df, ppm_tolerance, rt_tolerance, ccs_tolerance, eps_cutoff
    )
    print(df)
