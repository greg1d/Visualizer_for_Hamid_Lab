<<<<<<< HEAD
from combining_bins import process_and_combine

if __name__ == "__main__":
    df_final = process_and_combine(
        ms1_file="using_csv_no_rt/Low_only.csv",
        ms2_file="using_csv_no_rt/High_only.csv",
        output_location="test_sample 2.csv",
        beta=0.138218,
        tfix=-0.067817,
        min_mass=50,
        max_mass=2000,
        mass_bin_width=0.01,
        min_drift=0,
        max_drift=60,
        drift_bin_width=0.1,
        min_abundance=1000,
    )
    print(df_final)
=======
from combining_bins import process_data

if __name__ == "__main__":
    merged_df = process_data(
        ms1_file="using_csv_no_rt/Low_only.csv",
        ms2_file="using_csv_no_rt/High_only.csv",
        output_location="temp_sample.csv",
        beta=0.138218,
        tfix=-0.067817,
        combine_dfs=True,
        min_mass=50,
        max_mass=2000,
        mass_tolerance_ppm=10,
        min_ccs=0,
        max_ccs_factor=60,
        ccs_tolerance=0.02,
    )

    print(merged_df.head())
>>>>>>> 4be0c13785857ebcce08fc5603febad16e9d1dac
