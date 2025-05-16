from implementing_SQL import (
    exclude_noise_points,
    load_or_process_data,
    pick_a_cluster_for_debugging,
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

    # Load or process the data (caching with SQLite in .temp)
    df = load_or_process_data(
        file_path, tfix, beta, ppm_tolerance, rt_tolerance, ccs_tolerance
    )

    # Optionally exclude noise points
    df = exclude_noise_points(df, exclude_noise=exclude_noise_flag)

    # Debugging: Extract the largest cluster for analysis
    df = pick_a_cluster_for_debugging(df)

    # Calculate Cluster Relative Intensity
    df = calculate_cluster_relative_intensity(df)

    print("\nFinal DataFrame with Cluster Relative Intensities:")
    print(df)
