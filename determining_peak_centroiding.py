import warnings

# Suppress specific pyOpenMS OPENMS_DATA_PATH warning
warnings.filterwarnings(
    "ignore",
    message=".*OPENMS_DATA_PATH environment variable already exists.*",
    category=UserWarning,
)

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyopenms import *  # Must come after warning suppression  # noqa: E402, F403
from scipy.optimize import curve_fit  # noqa: E402

from determining_peak_profile import extract_cluster_by_number
from implementing_SQL import (  # noqa: E402
    exclude_noise_points,
    load_or_process_data,
)
from Open_MS_config import select_file  # noqa: E402


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


def gaussian(x, a, mu, sigma):
    """Gaussian function."""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def determine_mz_center(cluster_df):
    if cluster_df.empty:
        return None, None

    # Remove outliers in 'Base Peak m/z' using IQR method
    q1 = cluster_df["Base Peak m/z"].quantile(0.25)
    q3 = cluster_df["Base Peak m/z"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered_df = cluster_df[
        (cluster_df["Base Peak m/z"] >= lower_bound)
        & (cluster_df["Base Peak m/z"] <= upper_bound)
    ]

    if filtered_df.empty:
        return None, None

    filtered_df["rounded_mz"] = filtered_df["Base Peak m/z"].round(4)
    grouped = filtered_df.groupby("rounded_mz")["Cluster Relative Intensity"]

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
            filtered_df["Base Peak m/z"],
            weights=filtered_df["Cluster Relative Intensity"],
        )
        peak_intensity = filtered_df["Cluster Relative Intensity"].max()
        return weighted_mean, peak_intensity

    # Initial guesses for Gaussian fit
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
            filtered_df["Base Peak m/z"],
            weights=filtered_df["Cluster Relative Intensity"],
        )
        peak_intensity = filtered_df["Cluster Relative Intensity"].max()
        return weighted_mean, peak_intensity


def determine_dt_center(cluster_df):
    if cluster_df.empty:
        return None, None

    # Remove outliers in 'DT' using IQR method
    q1 = cluster_df["DT"].quantile(0.25)
    q3 = cluster_df["DT"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered_df = cluster_df[
        (cluster_df["DT"] >= lower_bound) & (cluster_df["DT"] <= upper_bound)
    ]

    if filtered_df.empty:
        return None, None

    filtered_df["rounded_dt"] = filtered_df["DT"].round(4)
    grouped = filtered_df.groupby("rounded_dt")["Cluster Relative Intensity"]

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
            filtered_df["DT"], weights=filtered_df["Cluster Relative Intensity"]
        )
        peak_intensity = filtered_df["Cluster Relative Intensity"].max()
        return weighted_mean, peak_intensity

    # Initial guesses for Gaussian fit
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
            filtered_df["DT"], weights=filtered_df["Cluster Relative Intensity"]
        )
        peak_intensity = filtered_df["Cluster Relative Intensity"].max()
        return weighted_mean, peak_intensity


def determine_rt_center(cluster_df):
    if cluster_df.empty:
        return None, None, None

    # Remove outliers in 'Retention Time (sec)' using IQR method
    q1 = cluster_df["Retention Time (sec)"].quantile(0.25)
    q3 = cluster_df["Retention Time (sec)"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered_df = cluster_df[
        (cluster_df["Retention Time (sec)"] >= lower_bound)
        & (cluster_df["Retention Time (sec)"] <= upper_bound)
    ]

    if filtered_df.empty:
        return None, None, None

    filtered_df["rounded_rt"] = filtered_df["Retention Time (sec)"].round(4)
    grouped = filtered_df.groupby("rounded_rt")["Cluster Relative Intensity"]

    try:
        stats = grouped.quantile(0.99).reset_index()
        stats["count"] = grouped.count().values
    except Exception:
        return None, None, None

    x_data = stats["rounded_rt"].values
    y_data = stats["Cluster Relative Intensity"].values
    weights = stats["count"].values

    if len(x_data) < 3:
        weighted_mean = np.average(
            filtered_df["Retention Time (sec)"],
            weights=filtered_df["Cluster Relative Intensity"],
        )
        peak_intensity = filtered_df["Cluster Relative Intensity"].max()
        spread = (
            filtered_df["Retention Time (sec)"].quantile(0.95)
            - filtered_df["Retention Time (sec)"].quantile(0.05)
        ) / 2
        return weighted_mean, peak_intensity, spread

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
        a, mu, sigma = popt
        return mu, a, sigma
    except RuntimeError:
        weighted_mean = np.average(
            filtered_df["Retention Time (sec)"],
            weights=filtered_df["Cluster Relative Intensity"],
        )
        peak_intensity = filtered_df["Cluster Relative Intensity"].max()
        spread = (
            filtered_df["Retention Time (sec)"].quantile(0.95)
            - filtered_df["Retention Time (sec)"].quantile(0.05)
        ) / 2
        return weighted_mean, peak_intensity, spread


def process_single_cluster(cluster_tuple):
    cluster_id, cluster_df = cluster_tuple
    cluster_df = calculate_cluster_relative_intensity(cluster_df)

    mz_center, amp_mz = determine_mz_center(cluster_df)
    dt_center, amp_dt = determine_dt_center(cluster_df)
    rt_center, amp_rt, sigma_rt = determine_rt_center(cluster_df)
    if "MS Level" in cluster_df.columns and not cluster_df["MS Level"].isnull().all():
        ms_level = cluster_df["MS Level"].mode().iloc[0]  # most common level
    else:
        ms_level = None
    return {
        "Cluster": cluster_id,
        "MS Level": ms_level,
        "m/z_ion_center": mz_center,
        "DT_center": dt_center,
        "Retention Time (sec)_center": rt_center,
        "Peak Intensity": cluster_df["Total Base Peak Intensity"].max(),
        "Amplitude_mz": amp_mz,
        "Amplitude_dt": amp_dt,
        "Amplitude_rt": amp_rt,
        "Sigma_rt": sigma_rt,
    }


def generate_cluster_centroid_report(df):
    """
    Parallelized generation of a simplified report with center and amplitude for each cluster.
    """
    cluster_ids = df["Cluster"].unique()
    cluster_data = [
        (cluster_id, df[df["Cluster"] == cluster_id].copy())
        for cluster_id in cluster_ids
    ]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_cluster, cluster_data))

    return pd.DataFrame(results)


def extract_cluster_by_mz_dt(
    df,
    mz_center=551.5,
    mz_tolerance=0.02,
    dt_center=33.0,
    dt_tolerance=0.1,
    mz_col="m/z_ion_center",
    dt_col="DT_center",
):
    # Check inputs
    if not isinstance(mz_col, str) or not isinstance(dt_col, str):
        raise ValueError("[ERROR] Column names must be strings.")

    if mz_col not in df.columns or dt_col not in df.columns:
        raise ValueError(
            f"[ERROR] Columns '{mz_col}' or '{dt_col}' not found in DataFrame."
        )

    condition = (df[mz_col].sub(mz_center).abs() <= mz_tolerance) & (
        df[dt_col].sub(dt_center).abs() <= dt_tolerance
    )

    return df[condition]


if __name__ == "__main__":
    tfix = -0.067817
    beta = 0.138218
    exclude_noise_flag = True

    # Example Data - Ensure this is a DataFrame
    file_path = select_file()
    ppm_tolerance = 1.5e-5
    rt_tolerance = 30
    ccs_tolerance = 0.02

    # Load or process the data (caching with SQLite in .temp)
    df = load_or_process_data(
        file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
    )

    # Optionally exclude noise points
    cluster_df = exclude_noise_points(df, exclude_noise=exclude_noise_flag)

    # Extract cluster by specified criteria
    cluster_df = calculate_cluster_relative_intensity(cluster_df)

    cluster_df_1 = extract_cluster_by_number(cluster_df, cluster_id=23437)
    cluster_df_2 = extract_cluster_by_number(cluster_df, cluster_id=23517)

    cluster_df_2 = generate_cluster_centroid_report(cluster_df_2)
    print(cluster_df_2)
