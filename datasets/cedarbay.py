import sys
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

# Add project root to sys.path before importing utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.tools import (
    get_images_from_directory,
    get_depth_maps_from_directory,
    get_file_id,
    load_config,
    load_depth_map,
)
from utils.view_depth import visualize_sample
import matplotlib.pyplot as plt
import cv2

def normalize_depth_map(depth_map: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    """
    Normalizes the depth map based on specified percentiles to enhance contrast.
    
    Parameters:
    - depth_map (np.ndarray): The original depth map.
    - lower_percentile (float): Lower percentile for normalization.
    - upper_percentile (float): Upper percentile for normalization.
    
    Returns:
    - np.ndarray: The normalized depth map.
    """
    lower = np.percentile(depth_map, lower_percentile)
    upper = np.percentile(depth_map, upper_percentile)
    depth_normalized = np.clip((depth_map - lower) / (upper - lower), 0, 1)
    return depth_normalized

class CedarBayDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        images_folder: str,
        depth_maps_folder: str,
        batch_size: int,
        target_height: int,
        target_width: int,
        crop_pixels: int = 0,
        shuffle: bool = True,
        pose_csv_path: str = None,
        pose_channels: int = 0
    ):
        """
        Initializes the CedarBayDataset.
        
        Parameters:
        - images_folder (str): Path to the images directory.
        - depth_maps_folder (str): Path to the depth maps directory.
        - batch_size (int): Number of samples per batch.
        - target_height (int): Height after resizing.
        - target_width (int): Width after resizing.
        - crop_pixels (int): Number of pixels to crop from the bottom.
        - shuffle (bool): Whether to shuffle the dataset each epoch.
        - pose_csv_path (str): Path to the pose data CSV file.
        - pose_channels (int): Number of pose channels to include (0, 1, or 3).
        """

        self.images_folder = images_folder
        self.depth_maps_folder = depth_maps_folder
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.crop_pixels = crop_pixels
        self.shuffle = shuffle
        self.pose_channels = pose_channels

        # Load file lists
        self.image_files = get_images_from_directory(images_folder)
        self.depth_files = get_depth_maps_from_directory(depth_maps_folder)

        # Ensure matching file IDs
        self.file_ids = [get_file_id(f) for f in self.image_files]
        self.depth_file_ids = [get_file_id(f, extension=".npy") for f in self.depth_files]

        if self.file_ids != self.depth_file_ids:
            raise ValueError("Image and depth map file IDs do not match.")

        # Load pose data if applicable
        if self.pose_channels > 0 and pose_csv_path:
            self.pose_df = pd.read_csv(pose_csv_path)
            self.pose_dict = self.pose_df.set_index('pose_timestamp').to_dict('index')
        else:
            self.pose_dict = {}

        self.indices = np.arange(len(self.image_files))
        self.on_epoch_end()

    def __len__(self) -> int:
        """Returns the number of batches per epoch."""
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates one batch of data.
        
        Parameters:
        - index (int): Batch index.
        
        Returns:
        - Tuple[np.ndarray, np.ndarray]: Batch of images and corresponding depth maps.
        """
        # Generate indices for the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        batch_depths = []

        for i in batch_indices:
            # File paths
            image_path = os.path.join(self.images_folder, self.image_files[i])
            depth_path = os.path.join(self.depth_maps_folder, self.depth_files[i])

            # Load image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.cast(image, tf.float32)

            # Load and decode depth map using NumPy
            depth_map = load_depth_map(depth_path)  # Returns NumPy array
            depth_map = normalize_depth_map(depth_map)  # Normalize for visualization
            depth_map = tf.convert_to_tensor(depth_map, dtype=tf.float32)

            # Preprocessing
            if self.crop_pixels > 0:
                image = image[:-self.crop_pixels, :, :]
                depth_map = depth_map[:-self.crop_pixels, :]

            # Preprocessing: Resize
            image = tf.image.resize(image, [self.target_height, self.target_width], method='bilinear')
            depth_map = tf.image.resize(depth_map[..., tf.newaxis], [self.target_height, self.target_width], method='bilinear')
            depth_map = tf.squeeze(depth_map, axis=-1)

            # Preprocessing: Normalize
            image = image / 255.0
            max_depth = tf.reduce_max(depth_map)
            depth_map = depth_map / max_depth if max_depth != 0 else depth_map

            # Incorporate pose data if applicable
            if self.pose_channels > 0:
                file_id = self.file_ids[i]
                pose_data = self.pose_dict.get(file_id, None)
                if pose_data:
                    if self.pose_channels == 1:
                        pose_channel = tf.constant([pose_data['relative_z']], dtype=tf.float32)
                    elif self.pose_channels == 3:
                        pose_channel = tf.constant([pose_data['relative_z'], pose_data['pitch'], pose_data['roll']], dtype=tf.float32)
                    else:
                        pose_channel = tf.constant([], dtype=tf.float32)
                    
                    # Expand dimensions to match (height, width, channels)
                    pose_channel = tf.tile(pose_channel, [self.target_height * self.target_width])
                    pose_channel = tf.reshape(pose_channel, [self.target_height, self.target_width, self.pose_channels])

                    # Concatenate pose channels to the image
                    image = tf.concat([image, pose_channel], axis=-1)
                else:
                    print(f"Pose data missing for file ID: {file_id}")

            batch_images.append(image.numpy())
            batch_depths.append(depth_map.numpy())

        return np.array(batch_images), np.array(batch_depths)

    def on_epoch_end(self):
        """Shuffles indices after each epoch if shuffle is enabled."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def main():
    """
    Testing functionality for CedarBayDataset.
    Loads a batch of data and visualizes sample images and depth maps.
    """
    # Define a test configuration dictionary
    test_config = {
        'dataset_parameters': {
            'images_folder': os.path.join('data', 'cedarbay', 'images'),
            'depth_maps_folder': os.path.join('data', "cedarbay", "depth_maps"),
            'image_height': 768,
            'image_width': 1024,
            'image_channels': 3,
            'depth_channels': 1,
            'batch_size': 8,
            'train_test_split': 0.75,
            'shuffle': True,
            'crop_pixels': 120,  # Based on our analysis
            'pose_csv_path': None,  # Not used for vanilla model
            'pose_channels_vanilla': 0  # Defined here for the vanilla model
        }
    }

    # Extract dataset parameters
    dataset_params = test_config['dataset_parameters']
    images_folder = dataset_params['images_folder']
    depth_maps_folder = dataset_params['depth_maps_folder']
    batch_size = dataset_params['batch_size']
    target_height = dataset_params['image_height']
    target_width = dataset_params['image_width']
    crop_pixels = dataset_params['crop_pixels']
    pose_csv_path = dataset_params.get('pose_csv_path', None)
    pose_channels = dataset_params.get('pose_channels_vanilla', 0)

    # Validate essential paths
    if not os.path.exists(images_folder):
        print(f"Images folder not found: {images_folder}")
        return
    if not os.path.exists(depth_maps_folder):
        print(f"Depth maps folder not found: {depth_maps_folder}")
        return

    # Instantiate the dataset for the vanilla model
    dataset = CedarBayDataset(
        images_folder=images_folder,
        depth_maps_folder=depth_maps_folder,
        batch_size=batch_size,
        target_height=target_height,
        target_width=target_width,
        crop_pixels=crop_pixels,
        shuffle=True,
        pose_csv_path=None,  # No pose data for vanilla model
        pose_channels=0
    )

    # Fetch a single batch
    try:
        images, depth_maps = dataset.__getitem__(0)
    except Exception as e:
        print(f"Error fetching batch: {e}")
        return
    
    # Visualize the first sample in the batch
    if len(images) > 0 and len(depth_maps) > 0:
        sample_image = images[0]
        sample_depth_map = depth_maps[0]

        # Convert image from [0,1] to [0,255] for visualization
        sample_image_vis = (sample_image * 255).astype(np.uint8)

        visualize_sample(
            image=sample_image_vis,
            depth_map=sample_depth_map,
            cmap='plasma',
            alpha=0.8,
            title_image="Sample RGB Image",
            title_depth="Sample Depth Map",
            title_overlay="Sample Depth Overlay"
        )
    else:
        print("No samples found in the batch.")


if __name__ == "__main__":
    main()