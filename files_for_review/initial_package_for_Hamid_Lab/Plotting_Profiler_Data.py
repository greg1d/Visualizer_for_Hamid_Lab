import pandas as pd
import numpy as np
import plotly.express as px
from openpyxl.utils import column_index_from_string


def filter_fixed_xls(input_path, output_path):
    df = pd.read_csv(input_path, sep="\t", skiprows=5)
    excel_cols_to_keep = ["A", "B", "D", "F", "H"]
    ai_index = 34  # AI = column 35 = index 34

    keep_indices = [column_index_from_string(col) - 1 for col in excel_cols_to_keep]
    keep_indices += [ai_index]  # only keep AI

    df_filtered = df.iloc[:, keep_indices]
    df_filtered.to_csv(output_path, index=False)
    print(f"[INFO] Filtered data saved to {output_path}")
    return df_filtered


def plot_2d(df, x_col, y_col, title, yaxis_label):
    if df is None or df.empty:
        print("[WARNING] No data to plot.")
        return

    df.columns = df.columns.astype(str)
    id_col = df.columns[0]
    rt_col = df.columns[1]
    ccs_col = df.columns[3]
    mz_col = df.columns[4]
    intensity_col = df.columns[5]  # AI

    df["log_intensity"] = np.log10(df[intensity_col] + 1)

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="log_intensity",
        color_continuous_scale="viridis",
        labels={
            mz_col: "m/z",
            ccs_col: "CCS (Drift Time ms)",
            rt_col: "Retention Time (sec)",
            "log_intensity": "log10(Intensity + 1)",
        },
        hover_data={
            id_col: True,
            rt_col: True,
            ccs_col: True,
            mz_col: True,
            intensity_col: True,
            "log_intensity": False,
        },
    )

    fig.update_layout(
        title=title,
        xaxis_title="m/z",
        yaxis_title=yaxis_label,
        xaxis_range=[0, 1650],
        yaxis_range=[0, 300],
        height=600,
        width=900,
    )

    fig.show()


if __name__ == "__main__":
    input_path = "data/test.xls"
    output_path = "data/test_filtered.csv"

    df_filtered = filter_fixed_xls(input_path, output_path)

    # Plot m/z vs CCS
    plot_2d(
        df_filtered,
        x_col=df_filtered.columns[4],  # m/z
        y_col=df_filtered.columns[3],  # CCS
        title="m/z vs CCS (colored by log10 AI intensity)",
        yaxis_label="CCS (Drift Time ms)",
    )

    # Plot m/z vs RT
    plot_2d(
        df_filtered,
        x_col=df_filtered.columns[4],  # m/z
        y_col=df_filtered.columns[1],  # RT
        title="m/z vs Retention Time (colored by log10 AI intensity)",
        yaxis_label="Retention Time (sec)",
    )
