import pandas as pd


def clean_csv(file_path, ms_mode):
    # Skip all rows until the line with 'Mass,Drift,Abundance'
    raw = pd.read_csv(
        file_path, header=0, skiprows=lambda x: x < 6
    )  # first 6 lines are metadata

    # Rename columns (just to be safe)
    raw.columns = ["Mass", "Drift", "Abundance"]

    # Convert to numeric and drop any bad rows
    for col in ["Mass", "Drift", "Abundance"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["Mass", "Drift", "Abundance"])

    # Add MS Mode
    raw["MS Mode"] = ms_mode
    return raw


def read_and_label(ms1_file=None, ms2_file=None, combine_dfs=True):
    dfs = []
    if ms1_file:
        ms1_df = clean_csv(ms1_file, 1)
        dfs.append(ms1_df)
    if ms2_file:
        ms2_df = clean_csv(ms2_file, 2)
        dfs.append(ms2_df)

    if not dfs:
        return pd.DataFrame()

    if combine_dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        # If not combining, just return MS1 (first dataframe)
        return dfs[0]


def calculate_CCS(df, beta, tfix):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if "Drift" not in df.columns or "Mass" not in df.columns:
        raise ValueError("DataFrame must contain 'Drift' and 'Mass' columns.")

    # Convert to numeric and coerce errors to NaN
    df["Mass"] = pd.to_numeric(df["Mass"], errors="coerce")
    df["Drift"] = pd.to_numeric(df["Drift"], errors="coerce")

    # Drop rows with NaN in Mass or Drift
    df = df.dropna(subset=["Mass", "Drift"])

    DT_gas = 28.006148  # gas mass in u
    gamma = (df["Mass"] / (DT_gas + df["Mass"])) ** 0.5
    adjusted_dt = df["Drift"] - tfix
    df["CCS"] = adjusted_dt / (beta * gamma)
    return df


if __name__ == "__main__":
    # User-defined parameters
    ms1_file = "using_csv_no_rt/Low_only.csv"
    ms2_file = "using_csv_no_rt/High_only.csv"
    tfix = -0.067817
    beta = 0.138218
    combine_dfs = True  # Set True to combine MS1+MS2, False to use MS1 only
    output_name = "xxxx_sample"  # User-specified name for output file

    # Read CSVs according to combine_dfs flag
    df = read_and_label(ms1_file, ms2_file, combine_dfs=combine_dfs)

    # Calculate CCS
    df = calculate_CCS(df, beta, tfix)

    # Save output
    output_file = f"using_csv_no_rt/{output_name}_with_CCS.csv"
    df.to_csv(output_file, index=False)
