import pandas as pd

from determining_peak_profile import (
    exclude_noise_points,
    load_or_process_data,
)
from Open_MS_config import select_file


def calculate_cluster_relative_intensity(df):
    """
    Calculate Cluster Relative Intensity for each row in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing cluster information and Base Peak Intensity.

    Returns:
    pd.DataFrame: DataFrame with an additional column for Cluster Relative Intensity.
    """
    if "Cluster" not in df.columns or "Base Peak Intensity" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'Cluster' and 'Base Peak Intensity' columns."
        )

    # Calculate max Base Peak Intensity per cluster
    cluster_max_intensity = df.groupby("Cluster")["Base Peak Intensity"].transform(
        "max"
    )
    total_base_peak_intensity = df.groupby("Cluster")[
        "Total Base Peak Intensity"
    ].transform("max")

    # Calculate Cluster Relative Intensity
    df["Cluster Relative Intensity"] = (
        df["Base Peak Intensity"] / cluster_max_intensity
    ) * total_base_peak_intensity

    return df


def calculate_weighted_centroids_for_clusters(df):
    """
    Calculate the weighted centroid (center) for each unique cluster in CCS, RT, DT, and m/z dimensions,
    weighted by the Cluster Relative Intensity.

    Parameters:
    df (pd.DataFrame): DataFrame containing the full dataset with cluster and intensity values.

    Returns:
    pd.DataFrame: A DataFrame with the weighted centroid values for each cluster.
    """
    if df.empty:
        print("No data in the DataFrame for centroid calculation.")
        return pd.DataFrame()

    # Ensure that the required columns exist
    required_columns = [
        "Cluster",
        "m/z_ion",
        "DT",
        "Retention Time (sec)",
        "CCS (Å^2)",
        "Cluster Relative Intensity",
        "Total Base Peak Intensity",
    ]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"DataFrame must contain the columns: {', '.join(required_columns)}"
        )

    # Calculate the weighted centroid for each cluster
    cluster_centroids = []

    for cluster_id, cluster_df in df.groupby("Cluster"):
        total_intensity = cluster_df["Cluster Relative Intensity"].sum()
        weighted_centroid = {
            "Cluster": cluster_id,
            "m/z_ion_center": (
                cluster_df["m/z_ion"] * cluster_df["Cluster Relative Intensity"]
            ).sum()
            / total_intensity,
            "DT_center": (
                cluster_df["DT"] * cluster_df["Cluster Relative Intensity"]
            ).sum()
            / total_intensity,
            "RT_center": (
                cluster_df["Retention Time (sec)"]
                * cluster_df["Cluster Relative Intensity"]
            ).sum()
            / total_intensity,
            "CCS_center": (
                cluster_df["CCS (Å^2)"] * cluster_df["Cluster Relative Intensity"]
            ).sum()
            / total_intensity,
            "Total Base Peak Intensity": cluster_df["Total Base Peak Intensity"].iloc[
                0
            ],  # Total intensity for the cluster
        }
        cluster_centroids.append(weighted_centroid)

    centroids_df = pd.DataFrame(cluster_centroids)

    print("\nWeighted Cluster Centroids for All Clusters:")
    print(centroids_df)

    return centroids_df


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
    df = calculate_cluster_relative_intensity(df)
    print(df)
    print("\nExtracted Cluster by Criteria:")
    df = calculate_weighted_centroids_for_clusters(df)
