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
