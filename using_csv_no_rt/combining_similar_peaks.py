import numpy as np
import pandas as pd
from centroiding_csv import generate_cluster_centroid_report
from clustering_csv import (
    calculate_CCS_for_csv_files,
    create_distance_matrix_sparse,
    perform_optimized_clustering,
)
from reading_in_csv import read_csv_with_numeric_header


def calculate_CCS_for_csv_files2(df, beta, tfix):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    DT_gas = 28.006148
    gamma = (df["m/z_ion_center"] / (DT_gas + df["m/z_ion_center"])) ** 0.5
    adjusted_dt = df["DT_center"] - tfix
    df["CCS (Å^2)"] = adjusted_dt / (beta * gamma)
    return df


def combine_similar_clusters(
    df,
    beta,
    tfix,
    ppm_tolerance=1.5e-5,
    ccs_tolerance=0.02,
):
    from scipy.spatial import KDTree

    if df.empty:
        return df

    df = df.reset_index(drop=True).copy()

    rename_map = {
        "m/z_ion_center": "m/z",
        "DT_center": "DT (ms)",
    }
    df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True
    )

    if df.empty:
        return df

    DT_gas = 28.006148
    gamma = (df["m/z"] / (DT_gas + df["m/z"])) ** 0.5
    adjusted_dt = df["DT (ms)"] - tfix
    df["CCS (Å^2)"] = adjusted_dt / (beta * gamma)

    used = set()
    combined_rows = []

    data = df[["m/z", "CCS (Å^2)"]].replace([np.inf, -np.inf], np.nan).dropna()
    valid_indices = data.index.to_list()
    df = df.loc[valid_indices].reset_index(drop=True)
    data = df[["m/z", "CCS (Å^2)"]].values
    tree = KDTree(data)

    for i in range(len(df)):
        if i in used:
            continue

        mz_i, ccs_i = data[i]
        mz_tol = mz_i * ppm_tolerance

        idxs = tree.query_ball_point([mz_i, ccs_i], r=np.inf)
        group = [
            j
            for j in idxs
            if j not in used
            and abs(data[j][0] - mz_i) <= mz_tol
            and abs(data[j][1] - ccs_i) <= ccs_tolerance
        ]

        used.update(group)
        group_df = df.iloc[group]

        avg_mz = group_df["m/z"].mean()
        avg_ccs = group_df["CCS (Å^2)"].mean()
        sum_intensity = group_df["Peak Intensity"].sum()

        combined_rows.append(
            {
                "m/z_ion_center": round(avg_mz, 4),
                "CCS (Å^2)": round(avg_ccs, 2),
                "Peak Intensity": sum_intensity,
            }
        )

    return pd.DataFrame(combined_rows)


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
    centroid_df = calculate_CCS_for_csv_files2(centroid_df, beta, tfix)
    print(centroid_df)

    combined_df = combine_similar_clusters(
        centroid_df, beta, tfix, ppm_tolerance, ccs_tolerance
    )
    print(combined_df)
    combined_df.to_csv("combined_clusters.csv", index=False)
