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
    pd.DataFrame: A DataFrame with clusters and their total Base Peak Intensity.
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


if __name__ == "__main__":
    # Example Data - Ensure this is a DataFrame
    file_path = select_file()
    exp = handle_uploaded_file(file_path)
    df = extract_spectrum_data(exp)

    df = pd.DataFrame(df)
    tfix = -0.067817
    beta = 0.138218
    df = calculate_CCS_for_mzML_files(df, beta, tfix)

    ppm_tolerance = 1e-5
    rt_tolerance = 30
    ccs_tolerance = 0.02

    sparse_matrix = create_distance_matrix_sparse(
        df, ppm_tolerance, rt_tolerance, ccs_tolerance
    )
    df = perform_optimized_clustering(df, sparse_matrix)

    # Sum Base Peak Intensity for each cluster
    df = sum_base_peak_intensity_per_cluster(df)

    print("\nFinal DataFrame with Peak Intensities:")
    print(df)
