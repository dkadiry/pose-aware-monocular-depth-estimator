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
    handle_infs_with_mask,
    normalize_depth_map_global,
    log_scale_depth_map
    )

from utils.view_depth import visualize_sample
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
import json

def create_gaussian_mask(height: int, width: int, sigma: float = None) -> np.ndarray:
    """
    Creates a 2D Gaussian mask.
    """
    if sigma is None:
        sigma = width / 6  # Common heuristic

    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xv, yv = np.meshgrid(x, y)
    d = np.sqrt(xv**2 + yv**2)
    gaussian = np.exp(-(d**2) / (2 * (sigma / width)**2))
    return gaussian

class CedarBayDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        images_folder: str,
        depth_maps_folder: str,
        batch_size: int,
        target_height: int,
        target_width: int,
        percentiles_path: str,
        pose_normalization_params_path: str,
        crop_pixels: int = 0,
        shuffle: bool = True,
        pose_csv_path: str = None,
        pose_channels: int = 0,
        subset_indices: List[int] = None
    ):
        """
        Initializes the CedarBayDataset.
        
        Parameters:
        - images_folder (str): Path to the images directory.
        - depth_maps_folder (str): Path to the depth maps directory.
        - batch_size (int): Number of samples per batch.
        - target_height (int): Height after resizing.
        - target_width (int): Width after resizing.
        - percentiles_path (str): Path to the percentiles npz
        - pose_normalization_params_path (str): Path to the pose normalization JSON file.
        - crop_pixels (int): Number of pixels to crop from the bottom.
        - shuffle (bool): Whether to shuffle the dataset each epoch.
        - pose_csv_path (str): Path to the pose data CSV file.
        - pose_channels (int): Number of pose channels to include (0, 1, 2, or 3).
        - subset_indices (List[int]): Indices to include in the dataset subset. (Used when generating specifically Training, Validation, and Test Sets)
        """

        self.images_folder = images_folder
        self.depth_maps_folder = depth_maps_folder
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.crop_pixels = crop_pixels
        self.shuffle = shuffle
        self.pose_channels = pose_channels
        self.percentiles_path = percentiles_path
        self.pose_normalization_params_path = pose_normalization_params_path

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

        # Load precomputed global percentiles
        if not os.path.exists(self.percentiles_path):
            raise FileNotFoundError(f"Percentiles file not found at: {self.percentiles_path}")
        percentiles = np.load(self.percentiles_path)
        self.lower_global = percentiles['P1']
        self.upper_global = percentiles['P99']
        print(f"Loaded Global Percentiles - P1: {self.lower_global}, P99: {self.upper_global}")

        # Load pose normalization parameters
        if self.pose_channels > 0:
            if not os.path.exists(self.pose_normalization_params_path):
                raise FileNotFoundError(f"Pose normalization params file not found at: {self.pose_normalization_params_path}")
            with open(self.pose_normalization_params_path, 'r') as json_file:
                self.pose_norm_params = json.load(json_file)
            print(f"Loaded Pose Normalization Parameters: {self.pose_norm_params}")
        else:
            self.pose_norm_params = {}

        # Handle subset indices
        if subset_indices is not None:
                self.indices = np.array(subset_indices)
        else:
            self.indices = np.arange(len(self.image_files))
        
        self.on_epoch_end()

    def __len__(self) -> int:
        """Returns the number of batches per epoch."""
        return  int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates one batch of data.
        Returns:
            Tuple containing:
                - Batch of masked images: np.ndarray
                - Batch of depth maps: np.ndarray
        """
        # Generate indices for the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        batch_depths = []
        #batch_masks = []

        for i, sample_idx in enumerate(batch_indices):
            try:
                # File paths
                image_path = os.path.join(self.images_folder, self.image_files[sample_idx])
                depth_path = os.path.join(self.depth_maps_folder, self.depth_files[sample_idx])

                # Load image
                image = tf.io.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.cast(image, tf.float32)

                # Load and decode depth map using NumPy
                depth_map_np = load_depth_map(depth_path)  # Returns NumPy array

                # Debug: Check raw depth map
                #print(f"Sample {sample_idx} - Raw Depth Map Shape: {depth_map_np.shape}")
                #print(f"Sample {sample_idx} - Raw Depth Map Min: {depth_map_np.min()}, Max: {depth_map_np.max()}")

                # Handle infs and get mask
                depth_map_np, mask = handle_infs_with_mask(depth_map_np)  

                # Debug: Check masked depth map
                #print(f"Sample {sample_idx} - Masked Depth Map Shape: {depth_map_np.shape}")
                #print(f"Sample {sample_idx} - Masked Depth Map Min: {depth_map_np.min()}, Max: {depth_map_np.max()}")

                # Normalize depth map to [0, 1] using global percentiles
                #depth_map_np = normalize_depth_map_global(depth_map_np, self.lower_global, self.upper_global)  

                # Apply logarithmic scaling to the depth map using global percentiles
                depth_map_np = log_scale_depth_map(depth_map_np, self.lower_global, self.upper_global)  # Log-scale normalization

                # Debug: Check normalized depth map
                #print(f"Sample {sample_idx} - Normalized Depth Map Min: {depth_map_np.min()}, Max: {depth_map_np.max()}")

                depth_map = tf.convert_to_tensor(depth_map_np, dtype=tf.float32)
                mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

                # Preprocessing: Crop
                if self.crop_pixels > 0:
                    image = image[:-self.crop_pixels, :, :]
                    depth_map = depth_map[:-self.crop_pixels, :]
                    mask_tensor = mask_tensor[:-self.crop_pixels, :]

                    # Debug: Check depth map after cropping
                    #print(f"Sample {sample_idx} - Cropped Depth Map Shape: {depth_map.shape}")
                    #print(f"Sample {sample_idx} - Cropped Depth Map Min: {depth_map.numpy().min()}, Max: {depth_map.numpy().max()}")

                # Preprocessing: Resize
                image = tf.image.resize(image, [self.target_height, self.target_width], method='bilinear')
                depth_map = tf.image.resize(depth_map[..., tf.newaxis], [self.target_height, self.target_width], method='bilinear')
                #depth_map = tf.squeeze(depth_map, axis=-1)
                mask_tensor = tf.image.resize(mask_tensor[..., tf.newaxis], [self.target_height, self.target_width], method='nearest')
                mask_tensor = tf.squeeze(mask_tensor, axis=-1)

                # Debug: Check depth map after resizing
                #print(f"Sample {sample_idx} - Resized Depth Map Shape: {depth_map.shape}")
                #print(f"Sample {sample_idx} - Resized Depth Map Min: {depth_map.numpy().min()}, Max: {depth_map.numpy().max()}")

                # Preprocessing: Normalize
                image = image / 255.0
                
                # Apply mask to the image: set invalid regions to zero
                image = tf.where(tf.expand_dims(mask_tensor, axis=-1) > 0, image, tf.zeros_like(image))

                # Debug: Check image before concantenating pose
                #print(f"Sample {sample_idx} - Image Shape before adding pose: {image.shape}")

                # Incorporate pose data if applicable
                if self.pose_channels > 0:
                    file_id = self.file_ids[sample_idx]
                    #print(f"Sample {sample_idx} - File/Pose Timestamp: {file_id}")
                    pose_data = self.pose_dict.get(file_id, None)
                    if pose_data:
                        pose_channels_encoded = self.encode_pose(pose_data, mask_tensor.numpy())
                        if pose_channels_encoded is not None:
                            # Concatenate pose channels to the image
                            image = tf.concat([image, pose_channels_encoded], axis=-1)
                            # Debug: Check image after concantenating pose
                            #print(f"Sample {sample_idx} - Image Shape after adding pose: {image.shape}")
                    else:
                        print(f"Pose data missing for file ID: {file_id}")
                
                batch_images.append(image.numpy())
                batch_depths.append(depth_map.numpy())
                #batch_masks.append(mask_tensor.numpy())


            except Exception as e:
                print(f"Error processing sample index {sample_idx}: {e}")

        return np.array(batch_images), np.array(batch_depths)

    def on_epoch_end(self):
        """Shuffles indices after each epoch if shuffle is enabled."""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_num_samples(self) -> int:
        """ Returns number of samples in the dataset """
        return len(self.indices)
    
    def encode_pose(self, pose_data: Dict[str, Any], mask: np.ndarray) -> tf.Tensor:
        """
        Encodes and normalizes pose data into image channels.
        
        Parameters:
        - pose_data (Dict[str, Any]): Dictionary containing pose values.
        - mask (np.ndarray): Mask array to apply to pose images.
        
        Returns:
        - tf.Tensor: Tensor containing encoded pose channels.
        """
        pose_channels = []
        pose_keys = []
        
        # Determine which pose keys to process based on pose_channels
        if self.pose_channels == 1:
            pose_keys = ['tz']
        elif self.pose_channels == 2:
            pose_keys = ['pitch', 'roll']
        elif self.pose_channels == 3:
            pose_keys = ['tz', 'pitch', 'roll']
        else:
            raise ValueError("pose_channels must be either 1, 2, or 3.")
        
        # Create Gaussian mask
        gaussian_mask = create_gaussian_mask(self.target_height, self.target_width, sigma=600)  # Adjust sigma as needed

        # Combine Gaussian mask with validity mask to ensure it only affects valid regions
        combined_mask = gaussian_mask * mask  # Element-wise multiplication

        for pose_key in pose_keys:
            if pose_key not in self.pose_norm_params:
                print(f"Normalization parameters for '{pose_key}' not found. Skipping this pose channel.")
                continue
            
            mean_val = self.pose_norm_params[pose_key]['mean']
            std_val = self.pose_norm_params[pose_key]['std']
            
            # Normalize the pose value to [mean=0, std=1]
            normalized_value = (pose_data[pose_key] - mean_val) / std_val

            # Clip the values to +-3 Std to handle any outliers
            normalized_value = np.clip(normalized_value, -3.0, 3.0)
            #print(f"Pre normalized pose value: {pose_data[pose_key]}, Post Normalized Pose value: {normalized_value}")
            
            # Create an image filled with the normalized pose value
            pose_image = np.full((self.target_height, self.target_width), normalized_value, dtype=np.float32)

            if pose_key == 'tz':
                # Apply Gaussian spatial modulation
                #pose_image *= combined_mask # Emphasize central 
                pose_image = np.where(mask > 0, pose_image, 0.0)
                
            if pose_key == 'pitch' or pose_key == 'roll':
                # Apply mask: set invalid regions to zero
                pose_image = np.where(mask > 0, pose_image, 0.0)
            
            # Optional: Apply Gaussian wsmoothing
            #pose_image = gaussian_filter(pose_image, sigma=1)  # Adjust sigma as needed
            
            # Expand dimensions to match (height, width, 1)
            pose_image = pose_image[..., np.newaxis]
            
            # Convert to tensor
            pose_tensor = tf.convert_to_tensor(pose_image, dtype=tf.float32)
            
            pose_channels.append(pose_tensor)
        
        if pose_channels:
            # Concatenate all pose channels along the last axis
            encoded_pose = tf.concat(pose_channels, axis=-1)
            return encoded_pose
        else:
            return None



class SubsetCedarBayDataset(CedarBayDataset):
    """Dataset subclass that represents a subset of the data."""

    def __init__(self, *args, subset_indices: List[int] = None, **kwargs):
        """
        Initializes the SubsetCedarBayDataset.

        Parameters:
        - subset_indices (List[int]): List of indices to include in the subset.
        - All other parameters are inherited from CedarBayDataset.
        """
        super().__init__(*args, **kwargs)
        if subset_indices is not None:
            self.indices = np.array(subset_indices)
        else:
            self.indices = np.arange(len(self.image_files))
        
        self.on_epoch_end()


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
            'pose_channels_vanilla': 0,  # Defined here for the vanilla model
            'percentiles_path': os.path.join('utils', 'percentiles.npz')
        }
    }

    # Load Config
    config_path = os.path.join("config", 'training_config.yaml')

    if not os.path.exists(config_path):
        print(f'Configuration file not found: {config_path}')
        return
    
    config = load_config(config_path)

    """

    # Extract dataset parameters using Test_config dictionary

    dataset_params = test_config['dataset_parameters']
    images_folder = dataset_params['images_folder']
    depth_maps_folder = dataset_params['depth_maps_folder']
    batch_size = dataset_params['batch_size']
    target_height = dataset_params['image_height']
    target_width = dataset_params['image_width']
    crop_pixels = dataset_params['crop_pixels']
    shuffle = dataset_params['shuffle']
    percentiles = dataset_params['percentiles_path']
    pose_csv_path = dataset_params.get('pose_csv_path', None)
    pose_channels = dataset_params.get('pose_channels_vanilla', 0)

    """
    # Extract dataset parameters using laoded training configuration
    dataset_params = config.get('dataset_parameters', {})
    images_folder = dataset_params.get('images_folder')
    depth_maps_folder = dataset_params.get('depth_maps_folder')
    batch_size = dataset_params.get('batch_size')
    target_height = dataset_params.get('image_height')
    target_width = dataset_params.get('image_width')
    crop_pixels = dataset_params.get('crop_pixels')
    shuffle = dataset_params.get('shuffle')
    percentiles = dataset_params.get('percentiles_path')
    pose_csv_path = dataset_params.get('pose_csv_path')
    pose_channels = dataset_params.get('pose_channels', {})
    pose_norm_path = dataset_params.get('pose_normalization_params_path')
    

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
        percentiles_path=percentiles,
        pose_normalization_params_path= pose_norm_path,
        crop_pixels=crop_pixels,
        shuffle=shuffle,
        pose_csv_path=pose_csv_path, 
        pose_channels=pose_channels['rel_z']        
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

        visualize_sample(
            image=sample_image,
            depth_map=sample_depth_map,
            mask=None,
            cmap='plasma',
            alpha=0.6,
            title_image="Sample Masked RGB Image",
            title_depth="Sample Normalized Depth Map",
            #title_mask="Sample Depth Mask",
            title_overlay="Sample Depth Overlay",
            mode = "all_with_pose"
        )
    else:
        print("No samples found in the batch.")


if __name__ == "__main__":
    main()