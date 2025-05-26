from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from clustering_csv import (
    calculate_CCS_for_csv_files,
    create_distance_matrix_sparse,
    perform_optimized_clustering,
)
from reading_in_csv import read_csv_with_numeric_header
from scipy.optimize import curve_fit


def gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def determine_mz_center(cluster_df):
    if cluster_df.empty:
        return None, None

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
    if (
        np.any(weights <= 0)
        or np.sum(weights) == 0
        or filtered_df["Abundance"].sum() == 0
    ):
        return filtered_df["Mass"].mean(), filtered_df["Abundance"].max()

    if len(x_data) < 3:
        return np.average(
            filtered_df["Mass"], weights=filtered_df["Abundance"]
        ), filtered_df["Abundance"].max()

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
        return popt[1], popt[0]
    except RuntimeError:
        if filtered_df["Abundance"].sum() == 0:
            return filtered_df["Mass"].mean(), 0
        return np.average(
            filtered_df["Mass"], weights=filtered_df["Abundance"]
        ), filtered_df["Abundance"].max()


def determine_dt_center(cluster_df):
    if cluster_df.empty:
        return None, None

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
    if (
        np.any(weights <= 0)
        or np.sum(weights) == 0
        or filtered_df["Abundance"].sum() == 0
    ):
        return filtered_df["Drift"].mean(), filtered_df["Abundance"].max()

    if len(x_data) < 3:
        return np.average(
            filtered_df["Drift"], weights=filtered_df["Abundance"]
        ), filtered_df["Abundance"].max()

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
        return popt[1], popt[0]
    except RuntimeError:
        if filtered_df["Abundance"].sum() == 0:
            return filtered_df["Drift"].mean(), 0
        return np.average(
            filtered_df["Drift"], weights=filtered_df["Abundance"]
        ), filtered_df["Abundance"].max()


def process_single_cluster(cluster_tuple):
    cluster_id, cluster_df = cluster_tuple
    cluster_df = cluster_df.copy()

    mz_center, amp_mz = determine_mz_center(cluster_df)
    dt_center, amp_dt = determine_dt_center(cluster_df)

    return {
        "Cluster": cluster_id,
        "m/z_ion_center": mz_center,
        "DT_center": dt_center,
        "Peak Intensity": cluster_df["Abundance"].max(),
        "Amplitude_mz": amp_mz,
        "Amplitude_dt": amp_dt,
    }


def generate_cluster_centroid_report(df):
    cluster_ids = df["Cluster"].unique()
    cluster_data = [
        (cluster_id, df[df["Cluster"] == cluster_id]) for cluster_id in cluster_ids
    ]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_cluster, cluster_data))

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = read_csv_with_numeric_header("hamid_labs_data/SB.csv")
    tfix = -0.067817
    beta = 0.138218
    df = calculate_CCS_for_csv_files(df, beta, tfix)

    ppm_tolerance = 1e-5
    ccs_tolerance = 0.02

    sparse_matrix = create_distance_matrix_sparse(df, ppm_tolerance, ccs_tolerance)
    df = perform_optimized_clustering(df, sparse_matrix)

    centroid_df = generate_cluster_centroid_report(df)
    centroid_df.to_csv("centroid_report.csv", index=False)
