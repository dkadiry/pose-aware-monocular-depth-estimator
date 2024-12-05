import tensorflow as tf
from models.unet import DepthEstimationModel, DownscaleBlock, UpscaleBlock, BottleNeckBlock
from datasets.cedarbay import CedarBayDataset, SubsetCedarBayDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
from utils.tools import seed_everything, load_config

from datetime import datetime  # Import datetime for timestamp generation

def get_run_identifier():
    """
    Generates a unique identifier based on the current timestamp.
    Format: YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    # Set seed for reproducibility
    seed_everything(42)

    # Load configuration
    config_path = 'config/training_config.yaml'
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

    # Select model variant to train ('vanilla', 'rel_z', or 'rel_z_pitch_roll')
    model_variant = experiment_params['model_variant']

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
        pose_channels=dataset_params['pose_channels'][model_variant]
    )

    print(f"Full dataset samples: {full_dataset.get_num_samples()} | Full dataset number of batches: {len(full_dataset)}")

    train_ratio = dataset_params['training_ratio']
    val_ratio = dataset_params['validation_ratio']
    test_ratio = dataset_params['test_ratio']

    train_indices_path = 'data/cedarbay/train_indices.npy'
    val_indices_path = 'data/cedarbay/val_indices.npy'
    test_indices_path = "data/cedarbay/test_indices.npy"

    # Check to see if dataset has been split before
    if os.path.exists(train_indices_path) and os.path.exists(val_indices_path) and os.path.exists(test_indices_path):
        train_indices = np.load(train_indices_path)
        val_indices = np.load(val_indices_path)
        test_indices = np.load(test_indices_path)
        print("Loaded existing train, validation, and test indices.")

    else:
        # Total number of samples
        total_samples = full_dataset.get_num_samples()
        print(f"Total samples in dataset pre-split: {total_samples}")

        # Create indices array
        indices = np.arange(total_samples)
        
        train_indices, temp_indices = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=42,
            shuffle=True
        )

        val_size_relative = val_ratio / (val_ratio + test_ratio)

        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_size_relative,
            random_state=42,
            shuffle=True
        )

        # Save split indices for reproducibility
        os.makedirs('data/cedarbay', exist_ok=True)
        np.save('data/cedarbay/train_indices.npy', train_indices)
        np.save('data/cedarbay/val_indices.npy', val_indices)
        np.save('data/cedarbay/test_indices.npy', test_indices)
        print(f"Saved train indices to {train_indices_path}, validation indices to {val_indices_path}, and test indices to '{test_indices_path}")

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
        pose_channels=dataset_params['pose_channels'][model_variant],
        subset_indices=train_indices.tolist()
    )

    val_dataset = SubsetCedarBayDataset(
        images_folder=dataset_params['images_folder'],
        depth_maps_folder=dataset_params['depth_maps_folder'],
        batch_size=dataset_params['batch_size'],
        target_height=dataset_params['image_height'],
        target_width=dataset_params['image_width'],
        percentiles_path=dataset_params['percentiles_path'],
        crop_pixels=dataset_params['crop_pixels'],
        shuffle=False,  
        pose_csv_path=dataset_params['pose_csv_path'],
        pose_channels=dataset_params['pose_channels'][model_variant],
        subset_indices=val_indices.tolist()
    )

    test_dataset = SubsetCedarBayDataset(
        images_folder=dataset_params['images_folder'],
        depth_maps_folder=dataset_params['depth_maps_folder'],
        batch_size=dataset_params['batch_size'],
        target_height=dataset_params['image_height'],
        target_width=dataset_params['image_width'],
        percentiles_path=dataset_params['percentiles_path'],
        crop_pixels=dataset_params['crop_pixels'],
        shuffle=False,  
        pose_csv_path=dataset_params['pose_csv_path'],
        pose_channels=dataset_params['pose_channels'][model_variant],
        subset_indices=test_indices.tolist()
    )
    
    print(f"Training samples: {train_dataset.get_num_samples()} | | Validation samples: {val_dataset.get_num_samples()} | Testing samples: {test_dataset.get_num_samples()}")
    print(f"Training batches: {len(train_dataset)} | Validation batches: {len(val_dataset)} |Testing batches: {len(test_dataset)}")

    assert len(np.intersect1d(train_indices, val_indices)) == 0, "Training and Validation sets overlap!"
    assert len(np.intersect1d(train_indices, test_indices)) == 0, "Training and Test sets overlap!"
    assert len(np.intersect1d(val_indices, test_indices)) == 0, "Validation and Test sets overlap!"

    #for images, depth_maps in train_dataset:
    #    print("Images shape:", images.shape)        # Expected: (batch_size, height, width, channels + pose_channels)
    #    print("Depth maps shape:", depth_maps.shape) # Expected: (batch_size, height, width, 1)
    #    break  # Only inspect the first batch

    # Initialize model vanilla, rel_z, or rel_z_pitch_roll
    input_shape = model_params['input_shapes'][model_variant]  # [768, 1024, 3]
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
        # Get weights from the pre-trained model
        pretrained_weights = pretrained_model.get_weights()
        # Get weights from the current model
        current_weights = model.get_weights()

        pretrained_model_input_channels = 3

        # If input channels differ, adjust the weights of the first Conv2D layer
        if model.input_channels != pretrained_model_input_channels:
            print("Adjusting weights for input channels...")
            # Verify weight shapes
            print(f"Pre-trained first Conv2D weights shape: {pretrained_weights[0].shape}")  # (kernel_h, kernel_w, in_channels, filters)
            print(f"Current first Conv2D weights shape: {current_weights[0].shape}")

            pretrained_first_layer_weights = pretrained_weights[0]
            pretrained_first_layer_bias = pretrained_weights[1]

            n_extra_channels = model.input_channels - pretrained_model_input_channels
            if n_extra_channels > 0:
                # Initialize new channel weights with small random values
                new_channels_shape = list(pretrained_first_layer_weights.shape)
                new_channels_shape[2] = n_extra_channels  # Adjust the input channel dimension
                new_channels_weights = np.random.normal(
                    loc=0.0, scale=0.01, size=new_channels_shape
                )

                # Concatenate along the input channel dimension
                adjusted_first_layer_weights = np.concatenate(
                    [pretrained_first_layer_weights, new_channels_weights],
                    axis=2
                )
            elif n_extra_channels < 0:
                # Slice the weights to match the reduced input channels
                adjusted_first_layer_weights = pretrained_first_layer_weights[:, :, :model.input_channels, :]
            else:
                # Input channels are the same
                adjusted_first_layer_weights = pretrained_first_layer_weights
                   
            # Replace the weights in the current model
            current_weights[0] = adjusted_first_layer_weights
            current_weights[1] = pretrained_first_layer_bias

            # Copy the rest of the weights
            current_weights[2:] = pretrained_weights[2:]
        
        else:
            # If input channels are the same, copy all weights
            current_weights = pretrained_weights
        
        # Set the weights to the current model
        model.set_weights(current_weights)
        print("Weights transferred successfully.")


    except Exception as e:
        print(f"Error transferring weights: {e}")
        return

    # Generate a unique run identifier
    run_id = get_run_identifier() 

    # Generate a unique run identifier
    run_id = get_run_identifier()  

    # Define the base log directory from configuration
    base_log_dir = logging_params['tensorboard_log_dir']  

    # Create a unique log directory for this run
    unique_log_dir = os.path.join(base_log_dir, f"run_{run_id}")

    # Ensure the directory exists
    os.makedirs(unique_log_dir, exist_ok=True)

    # Define Callbacks
    checkpoint_dir = experiment_params['models'][model_variant]['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_filepath = experiment_params['models'][model_variant]['checkpoint_file']
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=unique_log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )

    # Define callbacks list
    callbacks = [checkpoint_cb, early_stopping_cb, tensorboard_cb]
    
    """
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=experiment_params['epochs'],
        validation_data=val_dataset,  # Use validation set for monitoring
        callbacks=callbacks
    )

    # Save the final model
    saved_model_dir = experiment_params['models'][model_variant]['saved_model_dir']
    os.makedirs(saved_model_dir, exist_ok=True)
    model.save(saved_model_dir)
    print(f"Model saved to {saved_model_dir}")
    
    """
    

       
    

if __name__ == "__main__":
    main()