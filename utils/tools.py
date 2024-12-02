import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import yaml
from typing import List, Tuple, Dict, Any
import random
import tensorflow as tf

def rename_files_to_timestamps(folder_path: str) -> None:
    """
    Rename all .jpg and .txt files in the given folder to just their timestamps.
    
    The function assumes the filenames contain a timestamp after 'zed_image_left_' 
    and will rename the files to just the timestamp.
    
    Parameters:
    folder_path (str): The path to the folder containing the files to be renamed.
    """

    index = 1

    # Loop through each file and rename
    for filename in os.listdir(folder_path):
        
        if filename.endswith(".txt") or filename.endswith(".jpg") or filename.endswith(".npy"):
            # Extract timestamp
            timestamp = filename.split("depth_map_")[-1].split(".")[0]
            #timestamp = filename.split("_")[0]

            # Define newname
            file_ext = os.path.splitext(filename)[1]
            new_name = timestamp + file_ext
            
            # Define old and new path for rename function
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_name)

            # Rename File
            os.rename(old_file_path, new_file_path)
            print(f"Renamed file number {index}: {filename} -> {new_name}")
            index += 1

    print("All files renamed")

def collect_list_of_image_timestamps(folder_path: str) -> List[int]:
    index = 1
    timestamp_list = []

    for filename in os.listdir(folder_path):

        if filename.endswith('.jpg') or filename.endswith('.npy'):

            timestamp = int(filename.split('.')[0])
            timestamp_list.append(timestamp)

            #print(f"Timestamp {timestamp} collected from file {index} -> Name: {filename}")
            index +=1 

    return timestamp_list

def get_files_from_directory(directory: str, extensions: List[str]) -> List[str]:
    """
    Retrieves a sorted list of filenames from a directory filtered by extensions.
    
    Parameters:
    - directory (str): Path to the directory.
    - extensions (List[str]): List of allowed file extensions.
    
    Returns:
    - List[str]: Sorted list of matching filenames.
    """

    return sorted([
        filename for filename in os.listdir(directory) 
        if any(filename.lower().endswith(ext) for ext in extensions)
    ])

def get_images_from_directory(directory: str) -> List[str]:
    """
    Retrieves a sorted list of image filenames from a directory.
    
    Parameters:
    - directory (str): Path to the images directory.
    
    Returns:
    - List[str]: Sorted list of image filenames.
    """

    image_extensions = ['.jpg']
    return get_files_from_directory(directory, image_extensions)

def get_depth_maps_from_directory(directory: str) -> List[str]:
    """
    Retrieves a sorted list of depth map filenames from a directory.
    
    Parameters:
    - directory (str): Path to the depth maps directory.
    
    Returns:
    - List[str]: Sorted list of depth map filenames.
    """
    depth_map_extensions = ['.npy']
    return get_files_from_directory(directory, depth_map_extensions)

def get_file_id(filename: str, extension: str =".jpg") -> int:
    """
    Extracts the file ID from a filename.

    Parameters:
    - filename (str): The name of the file (e.g., "123456.jpg").
    - extension (str): The file extension to remove (default: '.jpg').

    Returns:
    - int: The extracted file ID.

    Raises:
    - ValueError: If the filename does not contain a valid integer ID.
    """

    file_id_str = filename.replace(extension, "")

    try:
        file_id = int(file_id_str)
        return file_id
    
    except ValueError:
        raise ValueError(f"Filename= '{filename}' does not contain a valid integer ID")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file

    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def create_directories(directories: List[str]) -> None:
    """
    Creates directories if they do not exist.

    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_depth_map(file_path: str) -> np.ndarray:
    """ 
    Loads Numpy Depth Map found at the path specified in the function parameter
    
    Raises:
        - FIleNotFoundError: If file does not exist
        - ValueError: If there was an error loading the file

    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Depth Map file not found: {file_path}")
    try:
        return np.load(file_path)
    except Exception as err:
        raise ValueError(f"Error loading depth map from {file_path}: {err}")

def save_depth_map(depth_map: np.ndarray, save_path: str) -> None:
    try:
        np.save(save_path, depth_map)
    except Exception as err:
        raise ValueError(f"Error saving depth map to {save_path}: {err}")


def associate_pose_with_depth_map(pose_df: pd.DataFrame, file_id: int) -> Dict[str, Any]:
    """
    Associates pose data with a specific depth map based on file ID.
    
    Parameters:
    - pose_df (pd.DataFrame): DataFrame containing pose data.
    - file_id (int): File ID corresponding to the depth map.
    
    Returns:
    - Dict[str, Any]: Associated pose parameters.
    
    Raises:
    - KeyError: If no matching pose data is found.
    """
    pose_row = pose_df[pose_df['pose_timestamp'] == file_id]
    if pose_row.empty:
        raise KeyError(f"No pose data found for file ID: {file_id}")
    return {
        'relative_z': pose_row['relative_z'].values[0],
        'pitch': pose_row['pitch'].values[0],
        'roll': pose_row['roll'].values[0]
    }

def handle_infs_with_mask(depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replaces infinite values in the depth map with zero and creates a mask.
    
    Parameters:
    - depth_map (np.ndarray): The original depth map.
    
    Returns:
    - Tuple[np.ndarray, np.ndarray]: The depth map with infs set to 0, and the mask.
    """
    # Create a mask where finite values are 1 and infs are 0
    mask = np.isfinite(depth_map).astype(np.float32)  # 1.0 for valid, 0.0 for inf
    
    # Find the highest row with an infinite value in each column
    inf_rows = np.where(~np.isfinite(depth_map), np.arange(depth_map.shape[0])[:, None], 0)
    max_inf_row = inf_rows.max(axis=0)

    # Extend the mask upwards for columns with inf values
    for col in range(mask.shape[1]):
        if max_inf_row[col] > 0:  # If there are inf values in the column
            mask[:max_inf_row[col], col] = 0.0  # Set all rows above the last inf to 0.0

    # Replace inf values with 0 for visualization
    depth_map = np.where(np.isfinite(depth_map), depth_map, 0.0)
    
    return depth_map, mask

def compute_global_percentiles(depth_maps_folder: str, lower: float = 1.0, upper: float = 99.0) -> Tuple[float, float]:
    """
    Computes global lower and upper percentiles across the dataset.

    Parameters:
    - depth_maps_folder (str): Path to the folder containing depth maps.
    - lower (float): Lower percentile (default: 1.0).
    - upper (float): Upper percentile (default: 99.0).

    Returns:
    - Tuple[float, float]: (P1, P99) percentile values.
    """
    all_finite_depths = []
    for depth_file in os.listdir(depth_maps_folder):
        if depth_file.endswith('.npy'):
            depth_map = load_depth_map(os.path.join(depth_maps_folder, depth_file))
            finite_depths = depth_map[np.isfinite(depth_map)]
            finite_depths = finite_depths[finite_depths > 0]  # Exclude zeros (replaced infs)
            all_finite_depths.extend(finite_depths.flatten())
    all_finite_depths = np.array(all_finite_depths)
    lower_percentile = np.percentile(all_finite_depths, lower)
    upper_percentile = np.percentile(all_finite_depths, upper)
    return lower_percentile, upper_percentile

def normalize_depth_map_global(depth_map: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Normalizes the depth map based on global percentiles.
    
    Parameters:
    - depth_map (np.ndarray): The original depth map.
    - lower (float): Global lower percentile value.
    - upper (float): Global upper percentile value.
    
    Returns:
    - np.ndarray: The normalized depth map in [0, 1].
    """
    # Normalize finite values
    depth_normalized = np.where(
        depth_map > 0,
        (depth_map - lower) / (upper - lower),
        0.0  # Keep zeros for inf regions
    )
    
    # Clip values to [0, 1]
    depth_normalized = np.clip(depth_normalized, 0.0, 1.0)
    
    return depth_normalized

def denormalize_depth_map_global(normalized_depth: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Denormalizes the depth map based on global percentiles.
    
    Parameters:
    - normalized_depth (np.ndarray): Normalized depth map in [0, 1].
    - lower (float): Global lower percentile value.
    - upper (float): Global upper percentile value.
    
    Returns:
    - np.ndarray: Denormalized depth map.
    """
    depth_denormalized = normalized_depth * (upper - lower) + lower
    return depth_denormalized

if __name__ == "__main__":
    pass
