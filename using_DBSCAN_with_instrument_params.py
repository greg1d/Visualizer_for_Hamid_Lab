import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, sort_graph_by_row_values
from sklearn.preprocessing import MinMaxScaler
from Open_MS_config import select_file, handle_uploaded_file, extract_spectrum_data


# Function to limit memory usage
def limit_memory_usage():
    """Calculates available memory to set a limit for processing."""
    mem = psutil.virtual_memory()
    available_memory = mem.available * 0.9  # Use 70% of available memory
    print(f"Available memory: {available_memory / (1024**2):.2f} MB")
    return available_memory


# Function to calculate m/z distance
def mz_distance(mz1, mz2, ppm_tolerance=1e-5):
    delta_mz = abs(mz2 - mz1)
    return delta_mz / (mz1 * ppm_tolerance)


# Function to calculate RT distance
def rt_distance(rt1, rt2, rt_tolerance=0.5):
    delta_rt = abs(rt2 - rt1)
    return delta_rt / rt_tolerance


# Function to calculate CCS distance
def ccs_distance(ccs1, ccs2, ccs_tolerance=0.02):
    delta_ccs = abs(ccs2 - ccs1)
    return delta_ccs / (ccs1 * ccs_tolerance)


# Create a sparse distance matrix
def create_distance_matrix_sparse(
    mz_values,
    rt_values,
    ccs_values,
    ppm_tolerance=1e-5,
    rt_tolerance=0.5,
    ccs_tolerance=0.02,
    eps_cutoff=None,  # Only store distances below this cutoff
):
    """
    Create a sparse distance matrix for large datasets.
    """
    mz_values = np.array(mz_values)
    rt_values = np.array(rt_values)
    ccs_values = np.array(ccs_values)

    n = len(mz_values)
    dist_matrix = csr_matrix((n, n), dtype=np.float32)  # Initialize sparse matrix
    for i in range(n):
        # Calculate distances for the current row
        mz_diff = np.abs(mz_values[i] - mz_values)
        rt_diff = np.abs(rt_values[i] - rt_values)
        ccs_diff = np.abs(ccs_values[i] - ccs_values)

        mz_dist = mz_diff / (mz_values[i] * ppm_tolerance)
        rt_dist = rt_diff / rt_tolerance
        ccs_dist = ccs_diff / (ccs_values[i] * ccs_tolerance)

        dist_row = np.sqrt(mz_dist**2 + rt_dist**2 + ccs_dist**2)

        # Store all distances except those greater than the cutoff in the sparse matrix
        if eps_cutoff is not None:
            below_cutoff = dist_row <= eps_cutoff
            dist_matrix[i, below_cutoff] = dist_row[below_cutoff]
        else:
            dist_matrix[i, :] = dist_row

    # Print the sparse matrix before converting to CSR format

    # Print non-zero elements of the sparse matrix
    print("Non-zero elements of the sparse distance matrix:")
    rows, cols = dist_matrix.nonzero()

    return dist_matrix.tocsr()  # Convert to Compressed Sparse Row format


def process_file(file_path):
    """
    Reads a .feather file and converts it to a NumPy array.
    """
    print(f"Processing file: {file_path}")
    try:
        df = pd.read_feather(file_path)

        # Ensure the columns exist and contain numeric data
        if all(col in df.columns for col in ["m/z", "Retention Time", "CCS"]):
            numeric_df = df[["m/z", "Retention Time", "CCS"]].apply(
                pd.to_numeric, errors="coerce"
            )
            data_array = (
                numeric_df.dropna().to_numpy()
            )  # Convert DataFrame to NumPy array, dropping rows with NaNs
            data_array = data_array[
                :3000000
            ]  # Limit to the first 20 features for debugging

            return data_array
        else:
            print("Required columns are missing in the DataFrame.")
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


# Read from the specified Feather file
file_paths = [
    ".temp/291 B4 16634.d.DeMP.feather",
    ".temp/295 B4 16707.d.DeMP.feather",
    ".temp/262 B4 MB-3.d.DeMP.feather",
    ".temp/261 B4 MB-2.d.DeMP.feather",
]
data_arrays = [process_file(file_path) for file_path in file_paths]
data_arrays = [data_array for data_array in data_arrays if data_array is not None]

if data_arrays:
    # Combine all data arrays into a single array
    combined_data_array = np.vstack(data_arrays)

    # Extract the m/z, RT, and CCS columns
    mz_values = combined_data_array[:, 0]
    rt_values = combined_data_array[:, 1]
    ccs_values = combined_data_array[:, 2]

    # Print the total number of features
    total_features = len(mz_values)
    print(f"Total number of features: {total_features}")

    # Parameters
    eps_cutoff = 1.732  # Adjusted EPS cutoff value for three dimensions
    ppm_tolerance = 1e-5
    rt_tolerance = 0.5
    ccs_tolerance = 0.02

    # Create distance matrix with memory limit
    available_memory = limit_memory_usage()
    memory_limit = available_memory * 0.9  # Use 10% of available memory

    # Create sparse distance matrix
    dist_matrix_sparse = create_distance_matrix_sparse(
        mz_values,
        rt_values,
        ccs_values,
        ppm_tolerance=ppm_tolerance,
        rt_tolerance=rt_tolerance,
        ccs_tolerance=ccs_tolerance,
        eps_cutoff=eps_cutoff,
    )

    # Sort the sparse matrix by row values
    dist_matrix_sparse_sorted = sort_graph_by_row_values(
        dist_matrix_sparse, warn_when_not_sorted=False
    )
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps_cutoff, min_samples=2, metric="precomputed")
    labels = dbscan.fit_predict(dist_matrix_sparse_sorted)

    # Print the number of clusters
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters: {num_clusters}")

    # Print values contained within each cluster
    for k in np.unique(labels):
        if k != -1:
            class_member_mask = labels == k
            cluster_mz = mz_values[class_member_mask]
            cluster_rt = rt_values[class_member_mask]
            cluster_ccs = ccs_values[class_member_mask]

    # Parameters for drift tolerance
    drift_mz_tolerance = 2 * ppm_tolerance
    drift_rt_tolerance = 2 * rt_tolerance
    drift_ccs_tolerance = 2 * ccs_tolerance

    # Refine clusters using drift tolerance
    for k in np.unique(labels):
        if k != -1:
            class_member_mask = labels == k
            cluster_mz = mz_values[class_member_mask]
            cluster_rt = rt_values[class_member_mask]
            cluster_ccs = ccs_values[class_member_mask]

            if len(cluster_mz) > 2 and len(cluster_rt) > 2 and len(cluster_ccs) > 2:
                # Calculate the core values of the cluster
                mz_core = np.percentile(cluster_mz, 50)
                rt_core = np.percentile(cluster_rt, 50)
                ccs_core = np.percentile(cluster_ccs, 50)
                # Dynamic drift tolerances
                dynamic_mass_drift = drift_mz_tolerance * mz_core
                dynamic_ccs_drift = drift_ccs_tolerance * ccs_core

                # Exclude points exceeding the drift tolerance
                drift_mask = (
                    (abs(cluster_mz - mz_core) <= dynamic_mass_drift)
                    & (abs(cluster_rt - rt_core) <= drift_rt_tolerance)
                    & (abs(cluster_ccs - ccs_core) <= dynamic_ccs_drift)
                )
                labels[class_member_mask] = np.where(drift_mask, k, -1)

    # Combine data into a single array
    points = np.vstack((mz_values, rt_values, ccs_values)).T

    # Scale data using Min-Max Scaler
    scaler = MinMaxScaler()
    points_scaled = scaler.fit_transform(points)

    # Calculate pairwise distances
    pairwise_distances = distance.cdist(
        points_scaled, points_scaled, metric="euclidean"
    )

    # Adjust radius using a meaningful percentile
    radius = np.percentile(
        pairwise_distances[pairwise_distances >= 0], 1
    )  # 1st percentile

    # Compute density using Nearest Neighbors
    nbrs = NearestNeighbors(radius=radius).fit(points_scaled)

    density = np.array(
        [len(nbrs.radius_neighbors([point])[0][0]) for point in points_scaled]
    )

    # Normalize density for coloring
    density_min = density.min()
    density_max = density.max()
    if density_max != density_min:
        density_normalized = (density - density_min) / (density_max - density_min)
    else:
        density_normalized = np.zeros_like(density)

    df = pd.DataFrame(
        {
            "m/z": mz_values,
            "RT": rt_values,
            "CCS": ccs_values,
            "Cluster": labels,
            "Density": density_normalized,
        }
    )

    cluster_colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "pink",
    ]
    df["Color"] = df["Cluster"].apply(
        lambda x: cluster_colors[x % len(cluster_colors)] if x != -1 else "grey"
    )

    # Option to color by density or cluster
    color_by = "density"  # Change to 'Cluster' to color by cluster

    # Plot with Plotly
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=df["m/z"],
            y=df["RT"],
            z=df["CCS"],
            mode="markers",
            marker=dict(
                size=2,
                color=df["Color"] if color_by == "Cluster" else df["Density"],
                colorscale="Viridis" if color_by == "Density" else None,
                colorbar=dict(title=color_by),
            ),
            text=df.apply(
                lambda row: f"m/z: {row['m/z']}, RT: {row['RT']}, CCS: {row['CCS']}, Cluster: {row['Cluster']}",
                axis=1,
            ),  # Hover text
            hoverinfo="text",
            name="Points",
        )
    )

    # Update layout
    fig.update_layout(
        scene=dict(xaxis_title="m/z", yaxis_title="RT", zaxis_title="CCS"),
        title="3D Scatter Plot with Cluster Binning Spanning Full Range",
    )

    fig.show()
else:
    print("Failed to process the file.")

if __name__ == "__main__":
    # Select and load the mzML file
    file_path = select_file()
    if not file_path:
        print("[ERROR] No file selected. Exiting.")
    else:
        exp = handle_uploaded_file(file_path)

        # Extract and display tabular data
        df = extract_spectrum_data(exp)
        print(df.head())
