from determining_peak_profile import (
    exclude_noise_points,
    extract_cluster_by_criteria,
    load_or_process_data,
)
from Open_MS_config import select_file

if __name__ == "__main__":
    # User-Editable Parameters
    tfix = -0.067817
    beta = 0.138218
    exclude_noise_flag = True

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

    # Extract cluster by specified criteria
    cluster_df = extract_cluster_by_criteria(
        df,
        mz_target=704.5227,
        dt_target=33.7,
        rt_target=300,
        mz_tolerance=0.1,
        dt_tolerance=1,
        rt_tolerance=50,
    )

    print("\nExtracted Cluster by Criteria:")
    print(cluster_df)
