import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict

def load_pose_data(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Loads pose data from a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file containing pose data.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing NumPy arrays of pose values for each component.
    """
    pose_components = ['tz', 'pitch', 'roll']
    pose_data = {component: np.array([]) for component in pose_components}

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return pose_data

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error: Failed to read CSV file {csv_path}: {e}")
        return pose_data

    for component in pose_components:
        if component in df.columns:
            # Convert to float and handle non-numeric entries
            pose_values = pd.to_numeric(df[component], errors='coerce')
            # Drop NaN values resulting from non-numeric entries
            pose_values = pose_values.dropna().values
            pose_data[component] = pose_values
            print(f"Loaded {len(pose_values)} valid entries for '{component}'.")
        else:
            print(f"Warning: '{component}' column not found in the CSV. This component will be skipped.")

    return pose_data

def compute_statistics(pose_data: Dict[str, np.ndarray], lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> Dict[str, Dict[str, float]]:
    """
    Computes mean and standard deviation for each pose component after excluding outliers.

    Parameters:
        pose_data (Dict[str, np.ndarray]): Dictionary containing NumPy arrays of pose values.
        lower_percentile (float): Lower percentile to exclude outliers.
        upper_percentile (float): Upper percentile to exclude outliers.

    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary with mean and std for each component.
    """
    stats = {}
    for component, values in pose_data.items():
        if values.size == 0:
            stats[component] = {'mean': None, 'std': None}
            print(f"Component: '{component}' | No data available to compute statistics.")
            continue

        # Compute lower and upper percentile values
        lower_bound = np.percentile(values, lower_percentile)
        upper_bound = np.percentile(values, upper_percentile)
        print(f"Component: '{component}' | {lower_percentile}th percentile: {lower_bound:.4f} | {upper_percentile}th percentile: {upper_bound:.4f}")

        # Filter values within the percentile bounds
        filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
        num_filtered = filtered_values.size
        print(f"Component: '{component}' | Values within {lower_percentile}-{upper_percentile}th percentile: {num_filtered}")

        if num_filtered == 0:
            stats[component] = {'mean': None, 'std': None}
            print(f"Component: '{component}' | No data within the specified percentile range. Skipping statistics computation.")
            continue

        # Compute mean and standard deviation on the filtered data
        mean = float(np.mean(filtered_values))
        std = float(np.std(filtered_values))
        stats[component] = {'mean': mean, 'std': std}
        print(f"Component: '{component}' | Mean (filtered): {mean:.4f} | Std (filtered): {std:.4f}\n")

    return stats

def save_statistics(stats: Dict[str, Dict[str, float]], output_path: str):
    """
    Saves the computed statistics to a JSON file.

    Parameters:
        stats (Dict[str, Dict[str, float]]): Nested dictionary with statistics.
        output_path (str): Path to the output JSON file.
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Pose statistics successfully saved to '{output_path}'.")
    except Exception as e:
        print(f"Error: Failed to save statistics to '{output_path}': {e}")

def main():
    """
    Main function to analyze pose data and compute statistics.
    """
  
    POSE_CSV_PATH = 'data\cedarbay\Filtered_Absolute_FFC_Pose_Euler.csv'          # Path to your pose CSV file
    OUTPUT_JSON_PATH = 'data\\cedarbay\\absolute_pose_stats.json'          # Desired output JSON file path

    # Percentile thresholds to exclude outliers
    LOWER_PERCENTILE = 0.0    # e.g., 1st percentile
    UPPER_PERCENTILE = 100.0   # e.g., 99th percentile
    

    print(f"Starting pose data analysis...\n")
    print(f"Pose CSV Path: {POSE_CSV_PATH}")
    print(f"Output JSON Path: {OUTPUT_JSON_PATH}")
    print(f"Excluding data below {LOWER_PERCENTILE}th percentile and above {UPPER_PERCENTILE}th percentile.\n")

    # Load pose data from CSV
    pose_data = load_pose_data(POSE_CSV_PATH)

    # Compute statistics excluding outliers
    stats = compute_statistics(pose_data, lower_percentile=LOWER_PERCENTILE, upper_percentile=UPPER_PERCENTILE)

    # Save the computed statistics to a JSON file
    save_statistics(stats, OUTPUT_JSON_PATH)

if __name__ == "__main__":
    main()
