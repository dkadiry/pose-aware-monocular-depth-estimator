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

if __name__ == "__main__":
    path = "data\cedarbay\images"
    depth_map_path = "data\cedarbay\depth_maps"
    
    poses_df = pd.read_csv("data\cedarbay\computed_FFC_relative_pose_data.csv")
    dmap_timestamps = collect_list_of_image_timestamps(depth_map_path)
    
    image_timestamps = collect_list_of_image_timestamps(path)
    
    pose_timestamp_series = poses_df['pose_timestamp'].astype('int64')
    pose_timestamp_set = set(pose_timestamp_series)
    im_timestamp_set = set(image_timestamps)
    dmap_set = set(dmap_timestamps)

    pose_set_type = {type(item) for item in pose_timestamp_set}
    im_set_type = {type(item) for item in im_timestamp_set}
    dmap_set_type = {type(item) for item in dmap_set}

    if pose_set_type == dmap_set_type and len(pose_set_type) == 1 and im_set_type == dmap_set_type:
        print("All sets have the same single data type.")

        missing_in_pose = dmap_set - pose_timestamp_set
        missing_in_dmap = pose_timestamp_set - dmap_set

        if missing_in_dmap or missing_in_pose:
            print(f'Missing in Pose timestamp but present in Dmap timestamps: {missing_in_pose}')
            print(f"Missing in Dmap timestamps, but present in pose timestamps: {missing_in_dmap}")

        else:
            print(f"Timestamps are aligned between datasets")

    else:
       print("Sets have either different or multiple data types")
