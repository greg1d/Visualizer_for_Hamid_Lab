import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from pyopenms import MSExperiment, MzMLFile


def select_file():
    """Opens a file dialog to select an mzML file and returns the file path."""
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Select an mzML File",
        filetypes=[("mzML Files", "*.mzML"), ("All Files", "*.*")],
    )
    return file_path


def handle_uploaded_file(file_path):
    """Reads the mzML file using pyOpenMS and returns an MSExperiment object."""
    exp = MSExperiment()

    if file_path:
        MzMLFile().load(file_path, exp)
        print(f"[INFO] Loaded mzML file: {file_path}")
        return exp
    else:
        print("[ERROR] No file selected.")
        return None


def extract_spectrum_data(exp):
    """Extracts Drift Time (DT), Retention Time (RT), and base peak m/z for the first 10 spectra."""
    drift_times = []
    retention_times = []
    mz_values = []

    for i, spectrum in enumerate(exp.getSpectra()[:10]):  # Only first 10 for now
        drift_time = spectrum.getDriftTime()
        rt = spectrum.getRT()

        mz_array, intensity_array = spectrum.get_peaks()
        mz = (
            mz_array[intensity_array.argmax()] if intensity_array.size > 0 else None
        )  # Base peak m/z

        # Debug
        print(f"[DEBUG] Spectrum {i + 1}: DT={drift_time}, RT={rt}, base m/z={mz}")

        drift_times.append(drift_time)
        retention_times.append(rt)
        mz_values.append(mz)

    df = pd.DataFrame(
        {
            "Spectrum Number": range(1, len(drift_times) + 1),
            "Drift Time (ms)": drift_times,
            "Retention Time (sec)": retention_times,
            "m/z": mz_values,
        }
    )

    print("\nFirst 10 spectra (Drift Time, Retention Time & base peak m/z):")
    print(df.to_string(index=False))

    return df


def plot_drift_time_vs_mz(df):
    """Plots Drift Time vs. m/z for the first 1000 spectra."""
    plt.figure(figsize=(10, 5))
    plt.scatter(df["m/z"], df["Drift Time (ms)"], color="red", marker="o")
    plt.xlabel("m/z")
    plt.ylabel("Drift Time (ms)")
    plt.title("Drift Time vs. m/z for First 1000 Spectra")
    plt.grid()
    plt.show()


def plot_retention_time_vs_mz(df):
    """Plots Retention Time vs. m/z for the first 1000 spectra."""
    plt.figure(figsize=(10, 5))
    plt.scatter(df["m/z"], df["Retention Time (sec)"], color="blue", marker="o")
    plt.xlabel("m/z")
    plt.ylabel("Retention Time (sec)")
    plt.title("Retention Time vs. m/z for First 1000 Spectra")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Select and load the mzML file
    file_path = select_file()
    if not file_path:
        print("[ERROR] No file selected. Exiting.")
    else:
        exp = handle_uploaded_file(file_path)

        # Extract and display tabular data
        df = extract_spectrum_data(exp)

        # Plot Drift Time vs. m/z
        plot_drift_time_vs_mz(df)

        # Plot Retention Time vs. m/z
        plot_retention_time_vs_mz(df)
