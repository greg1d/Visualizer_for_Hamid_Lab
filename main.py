from Open_MS_config import (
    plot_3d_mz_rt_intensity,
    plot_3d_mz_dt_intensity,
    extract_spectrum_data,
    handle_uploaded_file,
    select_file,
)
import pandas as pd


def run_peak_picking(df, mz_tol=0.01, rt_tol=2.0, dt_tol=1.0):
    """
    Cluster peaks and sum intensity within clusters.

    Parameters:
        df: DataFrame with 'm/z', 'Retention Time (sec)', 'Drift Time (ms)', 'Base Peak Intensity'
        mz_tol: m/z tolerance in Da
        rt_tol: retention time tolerance in seconds
        dt_tol: drift time tolerance in milliseconds
        intensity_thresh: only include points with intensity above this value

    Returns:
        A DataFrame of clustered peaks with summed intensity.
    """
    # Filter out low-intensity noise

    if df.empty:
        print("[WARNING] No data points above intensity threshold.")
        return pd.DataFrame(
            columns=[
                "m/z",
                "Retention Time (sec)",
                "Drift Time (ms)",
                "Intensity",
                "count",
            ]
        )

    # Scale for clustering
    scaled = df[["m/z", "Retention Time (sec)", "Drift Time (ms)"]].copy()
    scaled["m/z"] /= mz_tol
    scaled["Retention Time (sec)"] /= rt_tol
    scaled["Drift Time (ms)"] /= dt_tol

    # DBSCAN
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(eps=1.5, min_samples=2).fit(scaled)
    df["cluster"] = clustering.labels_

    # Keep only clustered points (exclude noise)
    clustered = df[df["cluster"] != -1]

    # Summarize
    summary = (
        clustered.groupby("cluster")
        .agg(
            **{
                "m/z": ("m/z", "mean"),
                "Retention Time (sec)": ("Retention Time (sec)", "mean"),
                "Drift Time (ms)": ("Drift Time (ms)", "mean"),
                "Intensity": ("Base Peak Intensity", "sum"),
                "count": ("Base Peak Intensity", "count"),
            }
        )
        .reset_index(drop=True)
    )
    summary = summary[summary["Intensity"] > 10000.0].reset_index(drop=True)

    return summary


if __name__ == "__main__":
    # Select and load the mzML file
    file_path = select_file()
    if not file_path:
        print("[ERROR] No file selected. Exiting.")
    else:
        exp = handle_uploaded_file(file_path)

        # Extract and display tabular data
        df = extract_spectrum_data(exp)

        peak_df = run_peak_picking(df, mz_tol=0.1, rt_tol=1.0, dt_tol=1.0)
        plot_3d_mz_rt_intensity(peak_df)
        plot_3d_mz_dt_intensity(peak_df)
        output_path_for_csv = "data/peak_picking_output.csv"
        peak_df.to_csv(output_path_for_csv, index=False)
