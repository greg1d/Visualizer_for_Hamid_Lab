import pandas as pd
from reading_in_csv import calculate_CCS, read_and_label

# ----------------------------
# Helper functions
# ----------------------------


def create_fixed_bins(min_val, max_val, tolerance, label_type="A"):
    """
    Create a reproducible list of bins between min_val and max_val.
    tolerance: absolute bin width
    Returns a DataFrame with 'Bin Lower', 'Bin Upper', 'Coordinate'
    """
    bins = []
    val = min_val
    coord_index = 0

    def number_to_letters(n):
        result = ""
        while True:
            n, remainder = divmod(n, 26)
            result = chr(65 + remainder) + result
            if n == 0:
                break
            n -= 1
        return result

    while val < max_val:
        lower = val
        upper = min(val + tolerance, max_val)
        bins.append((lower, upper, number_to_letters(coord_index)))
        val = upper
        coord_index += 1

    return pd.DataFrame(bins, columns=["Bin Lower", "Bin Upper", "Coordinate"])


def assign_coordinate(df, bin_df, col_name, min_val=None, max_val=None):
    """
    Assign each value in df[col_name] to a fixed bin from bin_df.
    Removes points outside min_val/max_val.
    """
    df = df.copy()

    if min_val is not None:
        df = df[df[col_name] >= min_val]
    if max_val is not None:
        df = df[df[col_name] <= max_val]

    coords = []
    bin_labels = []

    for val in df[col_name]:
        match = bin_df[(val >= bin_df["Bin Lower"]) & (val < bin_df["Bin Upper"])]
        if not match.empty:
            coords.append(match.iloc[0]["Coordinate"])
            bin_labels.append(
                f"{match.iloc[0]['Bin Lower']:.6f}–{match.iloc[0]['Bin Upper']:.6f}"
            )
        else:
            coords.append(None)
            bin_labels.append(None)

    df[f"{col_name} Coordinate"] = coords
    df[f"{col_name} Bin"] = bin_labels
    return df


# ----------------------------
# Main processing function
# ----------------------------


def process_mass_and_drift(
    ms1_file,
    ms2_file,
    output_location,
    beta,
    tfix,
    mass_tolerance=1.0,
    drift_tolerance=0.1,
    combine_dfs=True,
    min_mass=50,
    max_mass=1750,
):
    # Read CSVs
    df = read_and_label(ms1_file, ms2_file, combine_dfs=combine_dfs)

    # Calculate CCS values
    df = calculate_CCS(df, beta, tfix)

    # ----------------------------
    # Mass binning
    # ----------------------------
    mass_bins = create_fixed_bins(min_mass, max_mass, mass_tolerance)
    df = assign_coordinate(df, mass_bins, "Mass", min_val=min_mass, max_val=max_mass)

    # ----------------------------
    # Drift binning
    # ----------------------------
    min_drift = calculate_CCS(pd.DataFrame({"Mass": [1], "Drift": [tfix]}), beta, tfix)[
        "Drift"
    ].iloc[0]
    max_drift = calculate_CCS(
        pd.DataFrame({"Mass": [1], "Drift": [60 + tfix]}), beta, tfix
    )["Drift"].iloc[0]

    drift_bins = create_fixed_bins(min_drift, max_drift, drift_tolerance)
    df = assign_coordinate(
        df, drift_bins, "Drift", min_val=min_drift, max_val=max_drift
    )

    # Save output
    df.to_csv(output_location, index=False)
    print(f"Saved output to {output_location}")
    return df


# ----------------------------
# Main script
# ----------------------------

if __name__ == "__main__":
    ms1_file = "using_csv_no_rt/Low_only.csv"
    ms2_file = "using_csv_no_rt/High_only.csv"
    output_location = "mass_drift_binned_output.csv"
    beta = 0.138218
    tfix = -0.067817
    mass_tolerance = 10
    drift_tolerance = 0.1

    df = process_mass_and_drift(
        ms1_file=ms1_file,
        ms2_file=ms2_file,
        output_location=output_location,
        beta=beta,
        tfix=tfix,
        mass_tolerance=mass_tolerance,
        drift_tolerance=drift_tolerance,
    )
