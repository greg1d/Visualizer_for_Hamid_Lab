import hashlib
import os

import matplotlib.pyplot as plt
import pandas as pd

from implementing_SQL import (
    exclude_noise_points,
    pick_a_cluster_for_debugging,
)
from mzML_to_clustering_format import (
    calculate_CCS_for_mzML_files,
)
from Open_MS_config import extract_spectrum_data, handle_uploaded_file, select_file


# Ensure the .temp directory exists
def ensure_temp_directory():
    temp_dir = ".temp"
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def get_cache_file_path(
    file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
):
    """
    Generate a unique cache file path based on file name and parameters.
    """
    # Generate a unique hash for the file based on its path and processing parameters
    hash_input = (
        f"{file_path}_{tfix}_{beta}_{ppm_tolerance}_{rt_tolerance}_{ccs_tolerance}"
    )
    file_hash = hashlib.md5(hash_input.encode()).hexdigest()
    return os.path.join(ensure_temp_directory(), f"df_cache_{file_hash}.pkl")


def load_or_reprocess_data(
    file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
):
    """
    Load cached data if it exists, otherwise reprocess the data and cache it.
    """
    cache_file_path = get_cache_file_path(
        file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
    )

    # Check if the cached file exists
    if os.path.exists(cache_file_path):
        print(f"Loading cached data from {cache_file_path}...")
        df = pd.read_pickle(cache_file_path)
    else:
        print("Cached file not found or removed. Reprocessing data...")
        exp = handle_uploaded_file(file_path)
        df = extract_spectrum_data(exp)
        df = pd.DataFrame(df)

        # Apply calculations
        df = calculate_CCS_for_mzML_files(df, beta, tfix)

        # Save the processed data to the cache
        df.to_pickle(cache_file_path)
        print(f"Processed data cached to {cache_file_path}.")

    return df


def plot_3d_cluster(df):
    """
    Plot a 3D scatter plot for a selected cluster.

    Parameters:
    df (pd.DataFrame): DataFrame containing the cluster data with m/z_ion, CCS, and Cluster Relative Intensity.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot
    scatter = ax.scatter(
        df["m/z_ion"],
        df["CCS (Å^2)"],
        df["Cluster Relative Intensity"],
        c=df["Cluster Relative Intensity"],
        cmap="viridis",
        alpha=0.7,
        marker="o",
    )

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Cluster Relative Intensity")

    # Axis labels
    ax.set_xlabel("m/z_ion")
    ax.set_ylabel("CCS (Å^2)")
    ax.set_zlabel("Cluster Relative Intensity")
    ax.set_title("3D Plot of Cluster Relative Intensity")

    plt.show()


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

    # Load or reprocess data (cache or reprocess if removed)
    df = load_or_reprocess_data(
        file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
    )

    # Optionally exclude noise points
    df = exclude_noise_points(df, exclude_noise=exclude_noise_flag)

    # Debugging: Extract the 50th largest cluster for analysis
    df = pick_a_cluster_for_debugging(df)

    # Plot the 3D scatter plot
    plot_3d_cluster(df)
