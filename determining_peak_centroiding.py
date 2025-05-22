from determining_peak_profile import (
    exclude_noise_points,
    load_or_process_data,
)
from Open_MS_config import select_file

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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


import numpy as np


def gaussian(x, a, mu, sigma):
    """Gaussian function."""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def determine_mass_center(cluster_df):
    """
    Fit a Gaussian to the 99th percentile intensities at each m/z value,
    weighted by the number of values contributing to each m/z bin.
    Returns the peak center (mu) and peak intensity (amplitude a).
    """
    if cluster_df.empty:
        print("Empty cluster.")
        return None, None

    # Bin by rounding m/z
    cluster_df["rounded_mz"] = cluster_df["m/z_ion"].round(4)

    # Group by binned m/z, calculate 99th percentile and count of values
    grouped = cluster_df.groupby("rounded_mz")["Cluster Relative Intensity"]
    stats = grouped.quantile(0.99).reset_index()
    stats["count"] = grouped.count().values  # Weight = number of points in the bin

    x_data = stats["rounded_mz"].values
    y_data = stats["Cluster Relative Intensity"].values
    weights = stats["count"].values

    # Initial parameter guesses
    a_guess = np.max(y_data)
    mu_guess = x_data[np.argmax(y_data)]
    sigma_guess = (x_data.max() - x_data.min()) / 6

    try:
        # Weighted Gaussian curve fit
        popt, _ = curve_fit(
            gaussian,
            x_data,
            y_data,
            p0=[a_guess, mu_guess, sigma_guess],
            sigma=1 / weights,
            absolute_sigma=True,
        )
        a, mu, sigma = popt

        # Plotting
        x_fit = np.linspace(x_data.min(), x_data.max(), 500)
        y_fit = gaussian(x_fit, *popt)

        plt.figure(figsize=(8, 5))
        plt.scatter(
            x_data, y_data, s=weights * 2, alpha=0.7, label="99th percentile (weighted)"
        )
        plt.plot(x_fit, y_fit, color="red", label="Fitted Gaussian")
        plt.axvline(
            mu, color="green", linestyle="--", label=f"Peak center (μ={mu:.4f})"
        )
        plt.xlabel("m/z")
        plt.ylabel("Intensity")
        plt.title("Weighted Gaussian Fit of Cluster")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("Fitted Gaussian Parameters:")
        print(f"  Amplitude (a): {a:.4f}")
        print(f"  Center (μ): {mu:.4f}")
        print(f"  Std Dev (σ): {sigma:.4f}")

        return mu, a

    except RuntimeError:
        print("Gaussian fitting failed.")
        return None, None


def determine_dt_center(cluster_df):
    """
    Fit a Gaussian to the 99th percentile intensities at each m/z value,
    weighted by the number of values contributing to each m/z bin.
    Returns the peak center (mu) and peak intensity (amplitude a).
    """
    if cluster_df.empty:
        print("Empty cluster.")
        return None, None

    # Bin by rounding m/z
    cluster_df["rounded_dt"] = cluster_df["DT"].round(4)

    # Group by binned m/z, calculate 99th percentile and count of values
    grouped = cluster_df.groupby("rounded_dt")["Cluster Relative Intensity"]
    stats = grouped.quantile(0.99).reset_index()
    stats["count"] = grouped.count().values  # Weight = number of points in the bin

    x_data = stats["rounded_dt"].values
    y_data = stats["Cluster Relative Intensity"].values
    weights = stats["count"].values

    # Initial parameter guesses
    a_guess = np.max(y_data)
    mu_guess = x_data[np.argmax(y_data)]
    sigma_guess = (x_data.max() - x_data.min()) / 6

    try:
        # Weighted Gaussian curve fit
        popt, _ = curve_fit(
            gaussian,
            x_data,
            y_data,
            p0=[a_guess, mu_guess, sigma_guess],
            sigma=1 / weights,
            absolute_sigma=True,
        )
        a, mu, sigma = popt

        # Plotting
        x_fit = np.linspace(x_data.min(), x_data.max(), 500)
        y_fit = gaussian(x_fit, *popt)

        plt.figure(figsize=(8, 5))
        plt.scatter(
            x_data, y_data, s=weights * 2, alpha=0.7, label="99th percentile (weighted)"
        )
        plt.plot(x_fit, y_fit, color="red", label="Fitted Gaussian")
        plt.axvline(
            mu, color="green", linestyle="--", label=f"Peak center (μ={mu:.4f})"
        )
        plt.xlabel("DT")
        plt.ylabel("Intensity")
        plt.title("Weighted Gaussian with Weighted 99th percentile Fit of Cluster")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("Fitted Gaussian Parameters:")
        print(f"  Amplitude (a): {a:.4f}")
        print(f"  Center (μ): {mu:.4f}")
        print(f"  Std Dev (σ): {sigma:.4f}")

        return mu, a

    except RuntimeError:
        print("Gaussian fitting failed.")
        return None, None


def determine_rt_center(cluster_df):
    """
    Fit a Gaussian to the 99th percentile intensities at each m/z value,
    weighted by the number of values contributing to each m/z bin.
    Returns the peak center (mu) and peak intensity (amplitude a).
    """
    if cluster_df.empty:
        print("Empty cluster.")
        return None, None

    # Bin by rounding m/z
    cluster_df["rounded_rt"] = cluster_df["Retention Time (sec)"].round(4)

    # Group by binned m/z, calculate 99th percentile and count of values
    grouped = cluster_df.groupby("rounded_rt")["Cluster Relative Intensity"]
    stats = grouped.quantile(0.99).reset_index()
    stats["count"] = grouped.count().values  # Weight = number of points in the bin

    x_data = stats["rounded_rt"].values
    y_data = stats["Cluster Relative Intensity"].values
    weights = stats["count"].values

    # Initial parameter guesses
    a_guess = np.max(y_data)
    mu_guess = x_data[np.argmax(y_data)]
    sigma_guess = (x_data.max() - x_data.min()) / 6

    try:
        # Weighted Gaussian curve fit
        popt, _ = curve_fit(
            gaussian,
            x_data,
            y_data,
            p0=[a_guess, mu_guess, sigma_guess],
            sigma=1 / weights,
            absolute_sigma=True,
        )
        a, mu, sigma = popt

        # Plotting
        x_fit = np.linspace(x_data.min(), x_data.max(), 500)
        y_fit = gaussian(x_fit, *popt)

        plt.figure(figsize=(8, 5))
        plt.scatter(
            x_data, y_data, s=weights * 2, alpha=0.7, label="99th percentile (weighted)"
        )
        plt.plot(x_fit, y_fit, color="red", label="Fitted Gaussian")
        plt.axvline(
            mu, color="green", linestyle="--", label=f"Peak center (μ={mu:.4f})"
        )
        plt.xlabel("RT (sec)")
        plt.ylabel("Intensity")
        plt.title("Weighted Gaussian with Weighted 99th percentile Fit of Cluster")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("Fitted Gaussian Parameters:")
        print(f"  Amplitude (a): {a:.4f}")
        print(f"  Center (μ): {mu:.4f}")
        print(f"  Std Dev (σ): {sigma:.4f}")

        return mu, a

    except RuntimeError:
        print("Gaussian fitting failed.")
        return None, None


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
    determine_rt_center(cluster_df)
