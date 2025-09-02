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
