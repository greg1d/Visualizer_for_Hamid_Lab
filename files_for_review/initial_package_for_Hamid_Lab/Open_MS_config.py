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
    """Extracts DT, RT, base peak m/z, and base peak intensity for all spectra.
    Prints only those with m/z ∈ [861.45, 861.75] and DT ∈ [37.5, 42.5]. Also reports max RT.
    """
    drift_times = []
    retention_times = []
    mz_values = []
    base_peak_intensities = []

    max_rt = float("-inf")

    for i, spectrum in enumerate(exp.getSpectra()):
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

        drift_times.append(drift_time)
        retention_times.append(rt)
        mz_values.append(base_mz)
        base_peak_intensities.append(base_intensity)

        max_rt = max(max_rt, rt)

    df = pd.DataFrame(
        {
            "Spectrum Number": range(1, len(drift_times) + 1),
            "Drift Time (ms)": drift_times,
            "Retention Time (sec)": retention_times,
            "m/z": mz_values,
            "Base Peak Intensity": base_peak_intensities,
        }
    )

    # Filter based on m/z and Drift Time ranges
    filtered_df = df[
        (df["m/z"].between(0, 1600)) & (df["Drift Time (ms)"].between(0, 60))
    ].reset_index(drop=True)

    if not filtered_df.empty:
        print("\n[INFO] Filtered spectra:")
    else:
        print("[INFO] No spectra matched the specified m/z and DT criteria.")

    return filtered_df


def plot_3d_mz_dt_intensity(df):
    """Creates a 3D scatter plot of m/z, Drift Time, and Total Intensity."""
    if df.empty:
        print("[INFO] No data to plot in 3D.")
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        df["m/z"],
        df["Drift Time (ms)"],
        df["Intensity"],
        c=df["Intensity"],
        cmap="viridis",
        s=50,
    )

    ax.set_xlabel("m/z", labelpad=10)
    ax.set_ylabel("Drift Time (ms)", labelpad=10)
    ax.set_zlabel("Intensity", labelpad=10)
    ax.set_title("3D Plot: m/z vs Drift Time vs Total Base Peak Intensity")

    plt.tight_layout()
    plt.show()


def plot_3d_mz_rt_intensity(df):
    """Creates a 3D scatter plot of m/z, Drift Time, and Total Intensity."""
    if df.empty:
        print("[INFO] No data to plot in 3D.")
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        df["m/z"],
        df["Retention Time (sec)"],
        df["Intensity"],
        c=df["Intensity"],
        cmap="viridis",
        s=50,
    )

    ax.set_xlabel("m/z", labelpad=10)
    ax.set_ylabel("Retention Time (sec)", labelpad=10)
    ax.set_zlabel("Intensity", labelpad=10)
    ax.set_title("3D Plot: m/z vs Retention Time (sec) vs Total Base Peak Intensity")

    plt.tight_layout()
    plt.show()
