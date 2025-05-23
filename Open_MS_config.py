import tkinter as tk
from tkinter import filedialog

import pandas as pd
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
    """Extracts DT, RT, base peak m/z, and base peak intensity for all spectra.
    Prints MS level 1 and 2 spectra and specifically debugs the one with base peak m/z ≈ 551.5076.
    """
    drift_times = []
    retention_times = []
    mz_values = []
    base_peak_intensities = []
    ms_levels = []

    max_rt = float("-inf")

    print("\n[INFO] MS level 1 and 2 spectra:")

    for i, spectrum in enumerate(exp.getSpectra()):
        ms_level = spectrum.getMSLevel()
        drift_time = spectrum.getDriftTime()
        rt = spectrum.getRT()
        mz_array, intensity_array = spectrum.get_peaks()

        if intensity_array.size > 0:
            max_index = intensity_array.argmax()
            base_mz = mz_array[max_index]
            base_intensity = intensity_array[max_index]
        else:
            base_mz = None
            base_intensity = None

        # Record everything
        drift_times.append(drift_time)
        retention_times.append(rt)
        mz_values.append(base_mz)
        base_peak_intensities.append(base_intensity)
        ms_levels.append(ms_level)

        max_rt = max(max_rt, rt)

    df = pd.DataFrame(
        {
            "Spectrum Number": range(1, len(drift_times) + 1),
            "MS Level": ms_levels,
            "Drift Time (ms)": drift_times,
            "Retention Time (sec)": retention_times,
            "m/z": mz_values,
            "Base Peak Intensity": base_peak_intensities,
        }
    )

    return df


if __name__ == "__main__":
    # Select the mzML file
    file_path = select_file()

    # Handle the uploaded file
    exp = handle_uploaded_file(file_path)

    # Extract spectrum data
    if exp:
        df = extract_spectrum_data(exp)
        print(df)
    else:
        print("[ERROR] Failed to load mzML file.")
    df.to_csv("extracted_spectrum_data.csv", index=False)
