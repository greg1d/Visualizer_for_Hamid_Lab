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
        (df["m/z"].between(850.45, 861.75))
        & (df["Drift Time (ms)"].between(20.5, 42.5))
    ].reset_index(drop=True)

    if not filtered_df.empty:
        print("\n[INFO] Filtered spectra:")
    else:
        print("[INFO] No spectra matched the specified m/z and DT criteria.")

    return filtered_df


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


def collapse_close_mz_values(df):
    """
    Collapse rows where m/z values are within 1 ppm, and RT and DT are within 0.1 units.
    Returns a DataFrame with summed intensities and averaged RT/DT/mz.
    """
    if df.empty:
        print("[INFO] No data to group.")
        return df

    df_sorted = df.sort_values("m/z").reset_index(drop=True)
    grouped_data = []
    current_group = [df_sorted.iloc[0].to_dict()]

    for i in range(1, len(df_sorted)):
        current_row = df_sorted.iloc[i].to_dict()
        last_row = current_group[-1]

        ppm_error = abs((current_row["m/z"] - last_row["m/z"]) / last_row["m/z"])

        if ppm_error < 1e-6:
            current_group.append(current_row)
        else:
            # Collapse the current group
            total_intensity = sum(d["Base Peak Intensity"] for d in current_group)
            weighted_mz = (
                sum(d["m/z"] * d["Base Peak Intensity"] for d in current_group)
                / total_intensity
            )
            avg_rt = sum(d["Retention Time (sec)"] for d in current_group) / len(
                current_group
            )
            avg_dt = sum(d["Drift Time (ms)"] for d in current_group) / len(
                current_group
            )

            grouped_data.append(
                {
                    "m/z": weighted_mz,
                    "Total Intensity": total_intensity,
                    "Retention Time (sec)": avg_rt,
                    "Drift Time (ms)": avg_dt,
                }
            )

            current_group = [current_row]
    from statistics import mean

    # Final group
    if current_group:
        total_intensity = sum(d["Base Peak Intensity"] for d in current_group)
        weighted_mz = mean(d["m/z"] for d in current_group)

        avg_rt = sum(d["Retention Time (sec)"] for d in current_group) / len(
            current_group
        )
        avg_dt = sum(d["Drift Time (ms)"] for d in current_group) / len(current_group)

        grouped_data.append(
            {
                "m/z": weighted_mz,
                "Total Intensity": total_intensity,
                "Retention Time (sec)": avg_rt,
                "Drift Time (ms)": avg_dt,
            }
        )

    collapsed_df = pd.DataFrame(grouped_data)
    print(f"[INFO] Collapsed {len(df)} points into {len(collapsed_df)} grouped bins.")
    return collapsed_df


def plot_3d_mz_dt_intensity(df):
    """Creates a 3D scatter plot of m/z, Drift Time, and Total Intensity."""
    if df.empty:
        print("[INFO] No data to plot in 3D.")
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        df["m/z"],
        df["Retention Time (sec)"],
        df["Total Intensity"],
        c=df["Total Intensity"],
        cmap="viridis",
        s=50,
    )

    ax.set_xlabel("m/z", labelpad=10)
    ax.set_ylabel("Retention Time (sec)", labelpad=10)
    ax.set_zlabel("Total Intensity", labelpad=10)
    ax.set_title("3D Plot: m/z vs Drift Time vs Total Intensity")

    plt.tight_layout()
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
        collapsed_df = collapse_close_mz_values(df)

        collapsed_df = plot_3d_mz_dt_intensity(collapsed_df)
