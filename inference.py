# inference.py

import tensorflow as tf
from datasets.cedarbay import CedarBayDataset, SubsetCedarBayDataset
from models.unet import DepthEstimationModel, DownscaleBlock, UpscaleBlock, BottleNeckBlock
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
from utils.tools import denormalize_depth_map_global, seed_everything, load_config, save_depth_map
from utils.view_depth import visualize_and_save_inference_sample, save_error_map
import matplotlib.pyplot as plt

def main():
    # Set seed for reproducibility
    seed_everything(42)
    
    # Load inference configuration
    config_path = 'config/inference.yaml'
    if not os.path.exists(config_path):
        print(f"Inference configuration file not found: {config_path}")
        return
    
    config = load_config(config_path)
    
    # Extract parameters
    dataset_params = config['dataset_parameters']
    model_params = config['model_parameters']
    inference_params = config['inference_parameters']
    logging_params = config.get('logging_parameters', {})
    
    # Select model variant for inference
    model_variant = inference_params['model_variant']
    if model_variant not in model_params['input_shapes']:
        print(f"Invalid model_variant: {model_variant}. Available options: {list(model_params['input_shapes'].keys())}")
        return
    
    # Load split indices
    test_indices_path = 'data/cedarbay/test_indices.npy'
    if not os.path.exists(test_indices_path):
        print("Test indices file not found. Please run train.py first.")
        return
    test_indices = np.load(test_indices_path)
    
    # Initialize the test dataset
    test_dataset = SubsetCedarBayDataset(
        images_folder=dataset_params['images_folder'],
        depth_maps_folder=dataset_params['depth_maps_folder'],
        batch_size=dataset_params['batch_size'],
        target_height=dataset_params['image_height'],
        target_width=dataset_params['image_width'],
        percentiles_path=dataset_params['percentiles_path'],
        crop_pixels=dataset_params['crop_pixels'],
        shuffle=False,  # No shuffling for inference
        pose_csv_path=dataset_params['pose_csv_path'],
        pose_channels=dataset_params['pose_channels'][model_variant],
        subset_indices=test_indices.tolist()
    )
    
    print(f"Test dataset samples: {test_dataset.get_num_samples()} | Test dataset number of batches: {len(test_dataset)}")
    

    
    # Initialize model based on model_variant 'vanilla' 'rel_z' or 'rel_z_pitch_roll'
    input_shape = model_params['input_shapes'][model_variant] 
    input_channels = input_shape[2]
    model = DepthEstimationModel(
        width=input_shape[1],
        height=input_shape[0],
        input_channels=input_channels
    )
    
    # Build the model to initialize weights
    model.build(input_shape=(None, input_shape[0], input_shape[1], input_channels))
    model.summary()
    
    # Load the SavedModel
    saved_model_path = inference_params['models'][model_variant]['saved_model_path']
    if not os.path.exists(saved_model_path):
        print(f"SavedModel directory not found: {saved_model_path}")
        return
    
    try:
        model = tf.keras.models.load_model(saved_model_path)
        print(f"Loaded model from {saved_model_path}")
    except Exception as e:
        print(f"Error loading SavedModel: {e}")
        return
    
    # Load global percentiles
    if not os.path.exists(dataset_params['percentiles_path']):
        print(f"Percentiles file not found: {dataset_params['percentiles_path']}")
        return
    percentiles = np.load(dataset_params['percentiles_path'])
    lower_global = percentiles['P1']
    upper_global = percentiles['P99']
    
    # Create output directories
    output_dir = inference_params['models'][model_variant]['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize lists to store true and predicted depths
    all_true_depths = []
    all_pred_depths = []
    
    # Initialize lists to store error maps (optional)
    all_error_maps = []
    
    # Iterate over the test dataset and make predictions
    for batch_idx in range(len(test_dataset)):
        print(f"Processing batch {batch_idx + 1}/{len(test_dataset)}")
        images, depth_maps = test_dataset[batch_idx]
        
        # Make predictions
        predictions = model.predict(images)
        
        # Denormalize depth maps and predictions
        denorm_true = denormalize_depth_map_global(depth_maps, lower_global, upper_global)
        denorm_pred = denormalize_depth_map_global(predictions, lower_global, upper_global)
        
        # Append to the lists for metric computation
        all_true_depths.append(denorm_true)
        all_pred_depths.append(denorm_pred)
        
        # Iterate over each sample in the batch
        for i in range(images.shape[0]):
            true_depth = denorm_true[i]
            pred_depth = denorm_pred[i]
            
            # Generate error map
            error_map = np.abs(true_depth - pred_depth)
            all_error_maps.append(error_map)
            
            # Visualization and saving
            if inference_params.get('visualize', True):
                image = images[i]
                # Convert image from [0,1] to [0,255] for visualization - This is Only displayed for vanilla model as the pose and scale boosted model images have been concantenated with additional channels.
                image_vis = (image * 255).astype(np.uint8)
                save_path = os.path.join(output_dir, f"batch{batch_idx+1}_sample{i+1}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                visualize_and_save_inference_sample(
                    image=image_vis,
                    true_depth_map=true_depth,
                    pred_depth_map=pred_depth,
                    error_map=error_map,
                    save_path=save_path,
                    mode = model_variant                    
                )                

            if inference_params.get('save_predictions', True):
                # Save inidvidual prediction maps as .npy files

                # Define the filename
                pred_filename = f"pred_depth_map_batch{batch_idx+1}_sample{i+1}.npy"
                pred_path = os.path.join(output_dir, 'predictions', pred_filename)
                os.makedirs(os.path.dirname(pred_path), exist_ok=True)
                save_depth_map(pred_depth, pred_path)
                print(f"Saved predicted depth map to {pred_path}")
            
            if inference_params.get('save_error_maps', True):
                # Save error maps as .png files

                err_map_filename = f"error_map_batch{batch_idx+1}_sample{i+1}.png"
                err_map_path = os.path.join(output_dir, 'error_maps', err_map_filename)
                os.makedirs(os.path.dirname(err_map_path), exist_ok=True)
                save_error_map(error_map, err_map_path)
                print(f"Saved error map to {err_map_path}")

            # Compute Individual Metrics
            mse = mean_squared_error(true_depth, pred_depth)
            mae = mean_absolute_error(true_depth, pred_depth)
            rmse = np.sqrt(mean_squared_error(true_depth, pred_depth))

            # Save metrics to a file
            metrics_path = os.path.join(output_dir, f'evaluation_metrics{batch_idx+1}_sample{i+1}.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"Mean Squared Error (MSE): {mse}\n")
                f.write(f"Mean Absolute Error (MAE): {mae}\n")
                f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
            
            print(f"Saved evaluation metrics to {metrics_path}")
            
    
    # Concatenate all true and predicted depths for metric computation
    all_true_depths = np.concatenate(all_true_depths, axis=0).flatten()
    all_pred_depths = np.concatenate(all_pred_depths, axis=0).flatten()

    # Compute Metrics for All Predictions
    all_mse = mean_squared_error(all_true_depths, all_pred_depths)
    all_mae = mean_absolute_error(all_true_depths, all_pred_depths)
    all_rmse = np.sqrt(mean_squared_error(all_true_depths, all_pred_depths))

    # Save metrics to a file
    all_metrics_path = os.path.join(output_dir, f'evaluation_metrics{model_variant}.txt')
    with open(all_metrics_path, 'w') as f:
        f.write(f"Mean Squared Error (MSE): {all_mse}\n")
        f.write(f"Mean Absolute Error (MAE): {all_mae}\n")
        f.write(f"Root Mean Squared Error (RMSE): {all_rmse}\n")
    
    print(f"Mean Absolute Error All Depth Maps (MAE): {all_mae}")
    print(f"Mean Squared Error (MSE): {all_mse}")
    print(f"Root Mean Squared Error (RMSE): {all_rmse}")
    
    print("Inference and evaluation completed successfully.")
    
if __name__ == "__main__":
    main()
