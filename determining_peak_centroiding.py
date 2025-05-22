from determining_peak_profile import (
    exclude_noise_points,
    load_or_process_data,
)
from Open_MS_config import select_file
import pandas as pd


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


from scipy.optimize import curve_fit


def determine_mz_center(cluster_df):
    if cluster_df.empty:
        return None, None

    cluster_df["rounded_mz"] = cluster_df["m/z_ion"].round(4)
    grouped = cluster_df.groupby("rounded_mz")["Cluster Relative Intensity"]

    try:
        stats = grouped.quantile(0.99).reset_index()
        stats["count"] = grouped.count().values
    except Exception:
        return None, None

    x_data = stats["rounded_mz"].values
    y_data = stats["Cluster Relative Intensity"].values
    weights = stats["count"].values

    if len(x_data) < 3:
        weighted_mean = np.average(
            cluster_df["m/z_ion"], weights=cluster_df["Cluster Relative Intensity"]
        )
        peak_intensity = cluster_df["Cluster Relative Intensity"].max()
        return weighted_mean, peak_intensity

    a_guess = np.max(y_data)
    mu_guess = x_data[np.argmax(y_data)]
    sigma_guess = (x_data.max() - x_data.min()) / 6

    try:
        popt, _ = curve_fit(
            gaussian,
            x_data,
            y_data,
            p0=[a_guess, mu_guess, sigma_guess],
            sigma=1 / weights,
            absolute_sigma=True,
        )
        a, mu = popt[0], popt[1]
        return mu, a
    except RuntimeError:
        weighted_mean = np.average(
            cluster_df["m/z_ion"], weights=cluster_df["Cluster Relative Intensity"]
        )
        peak_intensity = cluster_df["Cluster Relative Intensity"].max()
        return weighted_mean, peak_intensity


def determine_dt_center(cluster_df):
    if cluster_df.empty:
        return None, None

    cluster_df["rounded_dt"] = cluster_df["DT"].round(4)
    grouped = cluster_df.groupby("rounded_dt")["Cluster Relative Intensity"]

    try:
        stats = grouped.quantile(0.99).reset_index()
        stats["count"] = grouped.count().values
    except Exception:
        return None, None

    x_data = stats["rounded_dt"].values
    y_data = stats["Cluster Relative Intensity"].values
    weights = stats["count"].values

    if len(x_data) < 3:
        weighted_mean = np.average(
            cluster_df["DT"], weights=cluster_df["Cluster Relative Intensity"]
        )
        peak_intensity = cluster_df["Cluster Relative Intensity"].max()
        return weighted_mean, peak_intensity

    a_guess = np.max(y_data)
    mu_guess = x_data[np.argmax(y_data)]
    sigma_guess = (x_data.max() - x_data.min()) / 6

    try:
        popt, _ = curve_fit(
            gaussian,
            x_data,
            y_data,
            p0=[a_guess, mu_guess, sigma_guess],
            sigma=1 / weights,
            absolute_sigma=True,
        )
        a, mu = popt[0], popt[1]
        return mu, a
    except RuntimeError:
        weighted_mean = np.average(
            cluster_df["DT"], weights=cluster_df["Cluster Relative Intensity"]
        )
        peak_intensity = cluster_df["Cluster Relative Intensity"].max()
        return weighted_mean, peak_intensity


def determine_rt_center(cluster_df):
    if cluster_df.empty:
        return None, None

    cluster_df["rounded_rt"] = cluster_df["Retention Time (sec)"].round(4)
    grouped = cluster_df.groupby("rounded_rt")["Cluster Relative Intensity"]

    try:
        stats = grouped.quantile(0.99).reset_index()
        stats["count"] = grouped.count().values
    except Exception:
        return None, None

    x_data = stats["rounded_rt"].values
    y_data = stats["Cluster Relative Intensity"].values
    weights = stats["count"].values

    if len(x_data) < 3:
        weighted_mean = np.average(
            cluster_df["Retention Time (sec)"],
            weights=cluster_df["Cluster Relative Intensity"],
        )
        peak_intensity = cluster_df["Cluster Relative Intensity"].max()
        return weighted_mean, peak_intensity

    a_guess = np.max(y_data)
    mu_guess = x_data[np.argmax(y_data)]
    sigma_guess = (x_data.max() - x_data.min()) / 6

    try:
        popt, _ = curve_fit(
            gaussian,
            x_data,
            y_data,
            p0=[a_guess, mu_guess, sigma_guess],
            sigma=1 / weights,
            absolute_sigma=True,
        )
        a, mu = popt[0], popt[1]
        return mu, a
    except RuntimeError:
        weighted_mean = np.average(
            cluster_df["Retention Time (sec)"],
            weights=cluster_df["Cluster Relative Intensity"],
        )
        peak_intensity = cluster_df["Cluster Relative Intensity"].max()
        return weighted_mean, peak_intensity


def generate_cluster_centroid_report(df):
    """
    Generate a simplified report with center and amplitude for each cluster.
    Returns a DataFrame where each row corresponds to a single cluster.
    """
    report_rows = []

    cluster_ids = df["Cluster"].unique()
    for cluster_id in cluster_ids:
        cluster_df = df[df["Cluster"] == cluster_id].copy()
        cluster_df = calculate_cluster_relative_intensity(cluster_df)

        mz_center, amp_mz = determine_mz_center(cluster_df)
        dt_center, amp_dt = determine_dt_center(cluster_df)
        rt_center, amp_rt = determine_rt_center(cluster_df)

        report_rows.append(
            {
                "Cluster": cluster_id,
                "m/z_ion_center": mz_center,
                "DT_center": dt_center,
                "Retention Time (sec)_center": rt_center,
                "Amplitude_mz": amp_mz,
                "Amplitude_dt": amp_dt,
                "Amplitude_rt": amp_rt,
            }
        )

    return pd.DataFrame(report_rows)


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

    # Exclude noise if specified
    df = exclude_noise_points(df, exclude_noise=exclude_noise_flag)

    # Generate simplified centroid report
    report_df = generate_cluster_centroid_report(df)

    # Output the report
    print("\nCluster Centroid Report:")
    print(report_df)
