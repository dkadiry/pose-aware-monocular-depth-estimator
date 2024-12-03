import yaml
import tensorflow as tf
from models.unet import DepthEstimationModel, DownscaleBlock, UpscaleBlock, BottleNeckBlock
from datasets.cedarbay import CedarBayDataset, SubsetCedarBayDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
from utils.tools import seed_everything, load_config

def main():
    # Set seed for reproducibility
    seed_everything(42)

    # Load configuration
    config_path = 'config/training.yaml'
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return
    
    config = load_config(config_path)

    # Extract parameters
    dataset_params = config['dataset_parameters']
    experiment_params = config['experiment_parameters']
    model_params = config['model_parameters']
    logging_params = config['logging_parameters']

    # Compute or load global percentiles
    if not os.path.exists(dataset_params['percentiles_path']):
        print("Percentiles file not found. Computing global percentiles...")
        from utils.tools import compute_global_percentiles
        lower, upper = compute_global_percentiles(dataset_params['depth_maps_folder'])
        np.savez(dataset_params['percentiles_path'], P1=lower, P99=upper)
        print(f"Computed and saved global percentiles: P1={lower}, P99={upper}")
    else:
        percentiles = np.load(dataset_params['percentiles_path'])
        lower = percentiles['P1']
        upper = percentiles['P99']
        print(f"Loaded Global Percentiles - P1: {lower}, P99: {upper}")

    # Initialize the full dataset
    full_dataset = CedarBayDataset(
        images_folder=dataset_params['images_folder'],
        depth_maps_folder=dataset_params['depth_maps_folder'],
        batch_size=dataset_params['batch_size'],
        target_height=dataset_params['image_height'],
        target_width=dataset_params['image_width'],
        percentiles_path=dataset_params['percentiles_path'],
        crop_pixels=dataset_params['crop_pixels'],
        shuffle=dataset_params['shuffle'],
        pose_csv_path=dataset_params['pose_csv_path'],
        pose_channels=dataset_params['pose_channels_vanilla']
    )

    # Check to see if dataset has been split before
    if os.path.exists('data/cedarbay/train_indices.npy') and os.path.exists('data/cedarbay/test_indices.npy'):
        train_indices = np.load('data/cedarbay/train_indices.npy')
        test_indices = np.load('data/cedarbay/test_indices.npy')

    else:
        # Split indices into train and test
        train_size = dataset_params['train_test_split']
        test_size = 1 - train_size
        random_seed = 42  # Ensure consistency
        train_indices, test_indices = train_test_split(
            np.arange(len(full_dataset)),
            train_size=train_size,
            test_size=test_size,
            random_state=random_seed,
            shuffle=True
        )

        # Save split indices for reproducibility
        os.makedirs('data/cedarbay', exist_ok=True)
        np.save('data/cedarbay/train_indices.npy', train_indices)
        np.save('data/cedarbay/test_indices.npy', test_indices)
        print(f"Saved train indices to 'data/cedarbay/train_indices.npy' and test indices to 'data/cedarbay/test_indices.npy'")

    # Create subset datasets
    train_dataset = SubsetCedarBayDataset(
        images_folder=dataset_params['images_folder'],
        depth_maps_folder=dataset_params['depth_maps_folder'],
        batch_size=dataset_params['batch_size'],
        target_height=dataset_params['image_height'],
        target_width=dataset_params['image_width'],
        percentiles_path=dataset_params['percentiles_path'],
        crop_pixels=dataset_params['crop_pixels'],
        shuffle=dataset_params['shuffle'],
        pose_csv_path=dataset_params['pose_csv_path'],
        pose_channels=dataset_params['pose_channels_vanilla'],
        subset_indices=train_indices.tolist()
    )

    test_dataset = SubsetCedarBayDataset(
        images_folder=dataset_params['images_folder'],
        depth_maps_folder=dataset_params['depth_maps_folder'],
        batch_size=dataset_params['batch_size'],
        target_height=dataset_params['image_height'],
        target_width=dataset_params['image_width'],
        percentiles_path=dataset_params['percentiles_path'],
        crop_pixels=dataset_params['crop_pixels'],
        shuffle=False,  # Typically no shuffling for test set
        pose_csv_path=dataset_params['pose_csv_path'],
        pose_channels=dataset_params['pose_channels_vanilla'],
        subset_indices=test_indices.tolist()
    )
    
    print(f"Training samples: {len(train_dataset)} | Testing samples: {len(test_dataset)}")

    # Initialize model vanilla, rel_z, or rel_z_pitch_roll
    input_shape = model_params['input_shapes']['vanilla']  # [768, 1024, 3]
    input_channels = input_shape[2]
    model = DepthEstimationModel(
        width=input_shape[1],
        height=input_shape[0],
        input_channels=input_channels
    )

    # Build the model to initialize weights
    model.build(input_shape=(None, input_shape[0], input_shape[1], input_shape[2]))
    model.summary()

    # Compile the model
    optimizer_name = experiment_params['optimizer'].lower()
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=experiment_params['learning_rate'])
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=experiment_params['learning_rate'])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
     # Compile the model
    model.compile(optimizer=optimizer)
    print("Model compiled successfully.")

    # Load the pre-trained model
    pretrained_model_path = 'saved_models/diode_model_unet_100_v4'

    try:
        pretrained_model = tf.keras.models.load_model(
            pretrained_model_path,
            custom_objects={
                'DepthEstimationModel': DepthEstimationModel,
                'DownscaleBlock': DownscaleBlock,
                'UpscaleBlock': UpscaleBlock,
                'BottleNeckBlock': BottleNeckBlock
            }
        )
        print("Pre-trained model loaded successfully.")
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return

    # Transfer weights
    try:
        model.set_weights(pretrained_model.get_weights())
        print("Weights transferred successfully.")
    except Exception as e:
        print(f"Error transferring weights: {e}")
        return
