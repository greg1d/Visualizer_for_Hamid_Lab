import pandas as pd

from determining_peak_profile import (
    exclude_noise_points,
    load_or_process_data,
)
from Open_MS_config import select_file
from pyopenms import MSSpectrum, PeakPickerHiRes

import matplotlib.pyplot as plt


from matplotlib.ticker import ScalarFormatter


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


def extract_cluster_for_debugging(df, cluster_id=0):
    """
    Extract a single cluster by ID for debugging.

    Parameters:
    df (pd.DataFrame): The full DataFrame with cluster annotations.
    cluster_id (int): The Cluster ID to extract.

    Returns:
    pd.DataFrame: Filtered DataFrame containing only the specified cluster.
    """
    if "Cluster" not in df.columns:
        raise ValueError("DataFrame must contain 'Cluster' column.")

    cluster_df = df[df["Cluster"] == cluster_id].copy()

    print(f"\nExtracted Cluster {cluster_id} for Debugging ({len(cluster_df)} rows):")
    print(cluster_df)

    return cluster_df


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

    return centroids_df


def pick_peaks_for_cluster_using_pyOpenMS(cluster_df):
    """
    Applies PeakPickerHiRes to a single cluster DataFrame.

    Parameters:
    cluster_df (pd.DataFrame): DataFrame containing m/z and intensity values.

    Returns:
    MSSpectrum: Picked spectrum after peak picking.
    """
    if cluster_df.empty:
        print("Empty cluster.")
        return None

    # Step 1: Convert to MSSpectrum
    spectrum = MSSpectrum()
    mz_array = cluster_df["m/z_ion"].values
    intensity_array = cluster_df["Cluster Relative Intensity"].values
    spectrum.set_peaks((mz_array, intensity_array))

    # Step 2: Apply PeakPickerHiRes
    picker = PeakPickerHiRes()
    picked_spectrum = MSSpectrum()
    picker.pick(spectrum, picked_spectrum)

    return picked_spectrum


import numpy as np


def pick_peaks_for_cluster_using_differentiation(cluster_df):
    """
    Plot a scatter plot of m/z vs Cluster Relative Intensity to visually inspect peak structure,
    and return the apex if a turning point is found via simple derivative analysis.

    Parameters:
    cluster_df (pd.DataFrame): DataFrame containing m/z and Cluster Relative Intensity.

    Returns:
    tuple: (apex_mz, apex_intensity) if a turning point is detected, otherwise (None, None)
    """
    if cluster_df.empty:
        print("Empty cluster.")
        return None, None

    # Sort by m/z for consistency
    sorted_df = cluster_df.sort_values(by="m/z_ion").reset_index(drop=True)
    mz_array = sorted_df["m/z_ion"].values
    intensity_array = sorted_df["Cluster Relative Intensity"].values

    # Plot m/z vs Cluster Relative Intensity
    plt.figure(figsize=(10, 5))
    plt.scatter(
        mz_array, intensity_array, color="blue", label="Cluster Relative Intensity"
    )
    plt.xlabel("m/z")
    plt.ylabel("Cluster Relative Intensity")
    plt.title("Scatter Plot: m/z vs Cluster Relative Intensity")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Optionally: also compute the turning point
    derivative = np.diff(intensity_array)
    for i in range(1, len(derivative)):
        if derivative[i - 1] > 0 and derivative[i] < 0:
            apex_mz = mz_array[i]
            apex_intensity = intensity_array[i]
            return apex_mz, apex_intensity

    print("No turning point found.")
    return None, None


def determine_line_equation_75th_percentile(cluster_df):
    """
    Computes a line equation (linear regression) using the 75th percentile of Cluster Relative Intensity
    for each unique m/z_ion value.

    Parameters:
    cluster_df (pd.DataFrame): DataFrame with 'm/z_ion' and 'Cluster Relative Intensity'.

    Returns:
    tuple: (slope, intercept)
    """
    if cluster_df.empty:
        print("Empty cluster.")
        return None, None

    # Group by m/z and take 75th percentile of intensity
    grouped = (
        cluster_df.groupby("m/z_ion")["Cluster Relative Intensity"]
        .apply(lambda x: np.percentile(x, 75))
        .reset_index(name="Intensity_75th")
    )

    x = grouped["m/z_ion"].values
    y = grouped["Intensity_75th"].values

    # Fit linear model y = mx + b
    slope, intercept = np.polyfit(x, y, deg=1)
    line_y = slope * x + intercept

    # Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color="blue", label="75th Percentile Intensities")
    plt.plot(
        x,
        line_y,
        color="red",
        linestyle="--",
        label=f"Fit: y = {slope:.2f}x + {intercept:.2f}",
    )
    plt.xlabel("m/z")
    plt.ylabel("75th Percentile Intensity")
    plt.title("Linear Fit Using 75th Percentile of Intensities at Each m/z")
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Line equation: y = {slope:.4f} * x + {intercept:.4f}")
    return slope, intercept


def gaussian(x, a, mu, sigma):
    """Gaussian function."""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


from scipy.optimize import curve_fit


def gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def fit_gaussian_to_75th_percentile(cluster_df):
    """
    Fit a Gaussian to the 75th percentile intensities at each m/z value,
    weighted by the number of values contributing to each m/z bin.
    Returns the peak apex (mu).
    """
    if cluster_df.empty:
        print("Empty cluster.")
        return None

    # Bin by rounding m/z
    cluster_df["rounded_mz"] = cluster_df["m/z_ion"].round(4)

    # Group by binned m/z, calculate 75th percentile and count of values
    grouped = cluster_df.groupby("rounded_mz")["Cluster Relative Intensity"]
    stats = grouped.quantile(0.95).reset_index()
    stats["count"] = grouped.count().values  # Weight = number of points in the bin

    x_data = stats["rounded_mz"].values
    y_data = stats["Cluster Relative Intensity"].values
    weights = stats["count"].values  # Higher count = more confidence

    # Initial parameter guesses
    a_guess = np.max(y_data)
    mu_guess = x_data[np.argmax(y_data)]
    sigma_guess = (x_data.max() - x_data.min()) / 6

    try:
        # Weighted curve fitting
        popt, _ = curve_fit(
            gaussian,
            x_data,
            y_data,
            p0=[a_guess, mu_guess, sigma_guess],
            sigma=1 / weights,
            absolute_sigma=True,
        )
        a, mu, sigma = popt

        # Plot
        x_fit = np.linspace(x_data.min(), x_data.max(), 500)
        y_fit = gaussian(x_fit, *popt)

        plt.figure(figsize=(8, 5))
        plt.scatter(
            x_data, y_data, s=weights * 2, label="95th percentile (weighted)", alpha=0.7
        )
        plt.plot(x_fit, y_fit, color="red", label="Fitted Gaussian")
        plt.axvline(
            mu, color="green", linestyle="--", label=f"Peak center (μ={mu:.4f})"
        )
        plt.xlabel("m/z")
        plt.ylabel("Intensity")
        plt.title("Gaussian Fit with Weighted 95th Percentile per m/z Bin")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("Fitted Gaussian Parameters (Weighted):")
        print(f"  Amplitude (a): {a:.4f}")
        print(f"  Center (μ): {mu:.4f} <-- dy/dx = 0 here")
        print(f"  Std Dev (σ): {sigma:.4f}")

        return mu

    except RuntimeError:
        print("Gaussian fitting failed.")
        return None


def plot_peak_overlay(cluster_df, picked_spectrum):
    """
    Plot the original base peak intensities and overlay the picked peaks.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Raw cluster data
    ax.scatter(
        cluster_df["m/z_ion"],
        cluster_df["Cluster Relative Intensity"],
        color="lightgray",
        label="Raw Peaks",
        s=10,
    )

    # Picked spectrum
    if picked_spectrum is not None:
        mzs, intensities = picked_spectrum.get_peaks()
        ax.scatter(
            mzs,
            intensities,
            color="red",
            label="Picked Peaks",
            s=25,
            marker="x",
        )

    # Axis and plot formatting
    ax.set_xlabel("m/z")
    ax.set_ylabel("Base Peak Intensity")
    ax.set_title("Peak Picking Overlay")
    ax.legend()

    # Force general number format on x-axis
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style="plain", axis="x")  # Disable scientific notation

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # User-Editable Parameters
    tfix = -0.067817
    beta = 0.138218
    exclude_noise_flag = True

    # Select file and set tolerances
    file_path = select_file()
    ppm_tolerance = 1e-5
    rt_tolerance = 30
    ccs_tolerance = 0.02

    # Load or process data
    df = load_or_process_data(
        file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
    )

    # Exclude noise and extract a specific cluster
    df = exclude_noise_points(df, exclude_noise=exclude_noise_flag)
    debug_cluster_id = 12051  # Specify the cluster ID to debug
    cluster_df = extract_cluster_for_debugging(df, cluster_id=debug_cluster_id)
    cluster_df = calculate_cluster_relative_intensity(cluster_df)
    determine_line_equation_75th_percentile(cluster_df)
    fit_gaussian_to_75th_percentile(cluster_df)

    # Apply peak picking algorithm using differentiation
    apex_mz, apex_intensity = pick_peaks_for_cluster_using_differentiation(cluster_df)
    if apex_mz is not None:
        print(
            f"\nPeak apex detected: m/z = {apex_mz:.4f}, Intensity = {apex_intensity:.2f}"
        )
