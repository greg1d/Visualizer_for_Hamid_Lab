import pandas as pd


def calculate_CCS_for_mzML_files(df, beta, tfix):
    """
    Calculate CCS for mzML files using the provided DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the spectrum data.
    beta (float): Beta value for CCS calculation - taken from IMBrowser (open Tune File in IMBrowser -> Integrate across whole RT range -> View -> CCS Calibration (Single Field) -> Select all mass peaks -> Find Drift Times -> Copy over mass values).
    tfix (float): Copy steps from getting the beta.

    Returns:
    pd.DataFrame: DataFrame with calculated CCS values.
    """
    # Ensure the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Rename columns for clarity
    df = df.rename(columns={"Drift Time (ms)": "DT", "m/z": "m/z_ion"})

    if "DT" not in df.columns or "m/z_ion" not in df.columns:
        raise ValueError("DataFrame must contain 'DT' and 'm/z_ion' columns.")

    # Constant for gas constant
    DT_gas = 28.006148

    # Calculate CCS using the EXACT specified formula
    gamma = (df["m/z_ion"] / (DT_gas + df["m/z_ion"])) ** 0.5
    print("Gamma values:", gamma)
    adjusted_dt = df["DT"] - tfix
    print("Adjusted Drift Time (ms):", adjusted_dt)
    df["CCS (Å^2)"] = adjusted_dt / (beta * gamma)
    return df


if __name__ == "__main__":
    # Example Data - Ensure this is a DataFrame
    data = {
        "Spectrum Number": [1, 2, 3],
        "Drift Time (ms)": [18.55, 14.0, 13.2],
        "Retention Time (sec)": [300, 320, 310],
        "m/z": [375.999, 510.1, 520.3],
        "Base Peak Intensity": [1000, 1200, 1100],
    }

    df = pd.DataFrame(data)
    tfix = -0.067817
    beta = 0.138218

    df_with_ccs = calculate_CCS_for_mzML_files(df, beta, tfix)
    print(df_with_ccs)
