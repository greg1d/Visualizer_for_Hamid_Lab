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
    """Extracts DT, RT, min m/z, base peak m/z, and base peak intensity for all spectra."""
    drift_times = []
    retention_times = []
    min_mz_values = []
    base_peak_mz_values = []
    base_peak_intensities = []
    ms_levels = []

    print("\n[INFO] MS level 1 and 2 spectra:")

    for i, spectrum in enumerate(exp.getSpectra()):
        ms_level = spectrum.getMSLevel()
        drift_time = spectrum.getDriftTime()
        rt = spectrum.getRT()
        mz_array, intensity_array = spectrum.get_peaks()

        if len(mz_array) > 0:
            min_mz = mz_array.min()
        else:
            min_mz = None

        if len(intensity_array) > 0:
            max_index = intensity_array.argmax()
            base_mz = mz_array[max_index]
            base_intensity = intensity_array[max_index]
        else:
            base_mz = None
            base_intensity = None

        drift_times.append(drift_time)
        retention_times.append(rt)
        min_mz_values.append(min_mz)
        base_peak_mz_values.append(base_mz)
        base_peak_intensities.append(base_intensity)
        ms_levels.append(ms_level)

    df = pd.DataFrame(
        {
            "Spectrum Number": range(1, len(drift_times) + 1),
            "MS Level": ms_levels,
            "Drift Time (ms)": drift_times,
            "Retention Time (sec)": retention_times,
            "Min m/z": min_mz_values,
            "Base Peak m/z": base_peak_mz_values,
            "Base Peak Intensity": base_peak_intensities,
        }
    )

    return df


if __name__ == "__main__":
    file_path = select_file()
    exp = handle_uploaded_file(file_path)

    if exp:
        df = extract_spectrum_data(exp)
        df.to_csv("extracted_spectrum_data.csv", index=False)
        print(df)
    else:
        print("[ERROR] Failed to load mzML file.")
