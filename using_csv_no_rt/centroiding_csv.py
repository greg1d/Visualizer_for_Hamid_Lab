import numpy as np
from clustering_csv import (
    calculate_CCS_for_csv_files,
    create_distance_matrix_sparse,
    perform_optimized_clustering,
)
from reading_in_csv import read_csv_with_numeric_header
from scipy.optimize import curve_fit


def gaussian(x, a, mu, sigma):
    """Gaussian function."""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def determine_mz_center(cluster_df):
    if cluster_df.empty:
        return None, None

    # IQR filtering on 'Mass'
    q1 = cluster_df["Mass"].quantile(0.25)
    q3 = cluster_df["Mass"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered_df = cluster_df[
        (cluster_df["Mass"] >= lower_bound) & (cluster_df["Mass"] <= upper_bound)
    ]

    if filtered_df.empty:
        return None, None

    filtered_df["rounded_mz"] = filtered_df["Mass"].round(4)
    grouped = filtered_df.groupby("rounded_mz")["Abundance"]

    try:
        stats = grouped.quantile(0.99).reset_index()
        stats["count"] = grouped.count().values
    except Exception:
        return None, None

    x_data = stats["rounded_mz"].values
    y_data = stats["Abundance"].values
    weights = stats["count"].values

    if len(x_data) < 3:
        weighted_mean = np.average(
            filtered_df["Mass"], weights=filtered_df["Abundance"]
        )
        peak_intensity = filtered_df["Abundance"].max()
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
            filtered_df["Mass"], weights=filtered_df["Abundance"]
        )
        peak_intensity = filtered_df["Abundance"].max()
        return weighted_mean, peak_intensity


def determine_dt_center(cluster_df):
    if cluster_df.empty:
        return None, None

    # IQR filtering on 'Drift'
    q1 = cluster_df["Drift"].quantile(0.25)
    q3 = cluster_df["Drift"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered_df = cluster_df[
        (cluster_df["Drift"] >= lower_bound) & (cluster_df["Drift"] <= upper_bound)
    ]

    if filtered_df.empty:
        return None, None

    filtered_df["rounded_dt"] = filtered_df["Drift"].round(4)
    grouped = filtered_df.groupby("rounded_dt")["Abundance"]

    try:
        stats = grouped.quantile(0.99).reset_index()
        stats["count"] = grouped.count().values
    except Exception:
        return None, None

    x_data = stats["rounded_dt"].values
    y_data = stats["Abundance"].values
    weights = stats["count"].values

    if len(x_data) < 3:
        weighted_mean = np.average(
            filtered_df["Drift"], weights=filtered_df["Abundance"]
        )
        peak_intensity = filtered_df["Abundance"].max()
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
            filtered_df["Drift"], weights=filtered_df["Abundance"]
        )
        peak_intensity = filtered_df["Abundance"].max()
        return weighted_mean, peak_intensity


if __name__ == "__main__":
    # Example Data - Ensure this is a DataFrame

    df = read_csv_with_numeric_header("hamid_labs_data/SB.csv")
    tfix = -0.067817
    beta = 0.138218
    df = calculate_CCS_for_csv_files(df, beta, tfix)

    ppm_tolerance = 1e-5
    ccs_tolerance = 0.02

    sparse_matrix = create_distance_matrix_sparse(df, ppm_tolerance, ccs_tolerance)
    df = perform_optimized_clustering(df, sparse_matrix)
    print(df.head())
    df = determine_dt_center(df)
    df = determine_mz_center(df)
