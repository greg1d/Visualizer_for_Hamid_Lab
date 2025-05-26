import pandas as pd


def read_csv_with_numeric_header(filepath):
    """
    Reads a CSV file and skips any initial non-data rows until a row starts with numeric values.
    Returns a cleaned DataFrame.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Detect the first row with numeric data
    for i, line in enumerate(lines):
        try:
            # Try parsing first value to float
            float(line.strip().split(",")[0])
            header_line = i - 1  # Header should be the line before numeric values
            break
        except ValueError:
            continue
    else:
        raise ValueError("No numeric data line found in the file.")

    # Now read using pandas starting at the detected header
    df = pd.read_csv(filepath, skiprows=header_line)
    return df


if __name__ == "__main__":
    df = read_csv_with_numeric_header("hamid_labs_data/SB.csv")
    print(df.head())
