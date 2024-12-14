import os
import pandas as pd
import json
from typing import List

def analyze_pose_data(pose_csv_path: str, output_json_path: str, pose_columns: List[str]):
    """
    Analyzes the pose CSV file to find min and max values for specified pose columns
    and saves the statistics to a JSON file.
    
    Parameters:
    - pose_csv_path (str): Path to the pose CSV file.
    - output_json_path (str): Path where the JSON file with statistics will be saved.
    - pose_columns (List[str]): List of pose columns to analyze (e.g., ['tz', 'pitch', 'roll']).
    """
    if not os.path.exists(pose_csv_path):
        raise FileNotFoundError(f"Pose CSV file not found at: {pose_csv_path}")
    
    # Load pose data
    pose_df = pd.read_csv(pose_csv_path)
    
    # Initialize dictionary to hold min and max values
    pose_stats = {}
    
    for col in pose_columns:
        if col not in pose_df.columns:
            print(f"Warning: Column '{col}' not found in pose CSV. Skipping.")
            continue
        min_val = pose_df[col].min()
        max_val = pose_df[col].max()
        pose_stats[col] = {
            'min': float(min_val),
            'max': float(max_val)
        }
        print(f"Pose Column '{col}': min = {min_val}, max = {max_val}")
    
    # Save statistics to JSON
    with open(output_json_path, 'w') as json_file:
        json.dump(pose_stats, json_file, indent=4)
    
    print(f"\nPose statistics saved to '{output_json_path}'.")

if __name__ == "__main__":
    
    # Define paths
    pose_csv_path = 'data/cedarbay/Filtered_Absolute_FFC_Pose_Euler.csv' 
    output_json_path = 'data/cedarbay/absolute_pose_normalization_params.json'  
    pose_columns = ['tz', 'pitch', 'roll'] 
    
    analyze_pose_data(pose_csv_path, output_json_path, pose_columns)
