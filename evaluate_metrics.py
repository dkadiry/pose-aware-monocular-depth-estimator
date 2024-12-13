import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.tools import load_config


def find_metric_files(metrics_dir):
    """
    Finds all per-sample metric .txt files in the specified directory.
    
    Parameters:
    - metrics_dir (str): Directory where metric files are stored.
    
    Returns:
    - list: List of file paths matching the pattern.
    """
    # Pattern to match all per-sample metric files
    pattern = os.path.join(metrics_dir, 'evaluation_metrics*_sample*.txt')
    metric_files = glob.glob(pattern, recursive=True)
    return metric_files

def parse_metric_file(file_path):
    """
    Parses a single metric file and extracts MSE, MAE, and RMSE.
    
    Parameters:
    - file_path (str): Path to the metric .txt file.
    
    Returns:
    - dict: Dictionary containing MSE, MAE, and RMSE.
    """
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            if 'Mean Squared Error (MSE):' in line:
                try:
                    metrics['MSE'] = float(line.strip().split(':')[-1])
                except ValueError:
                    print(f"Warning: Could not parse MSE in file {file_path}")
            elif 'Mean Absolute Error (MAE):' in line:
                try:
                    metrics['MAE'] = float(line.strip().split(':')[-1])
                except ValueError:
                    print(f"Warning: Could not parse MAE in file {file_path}")
            elif 'Root Mean Squared Error (RMSE):' in line:
                try:
                    metrics['RMSE'] = float(line.strip().split(':')[-1])
                except ValueError:
                    print(f"Warning: Could not parse RMSE in file {file_path}")
    return metrics

def aggregate_metrics(metric_files):
    """
    Aggregates metrics from multiple files into a DataFrame.
    
    Parameters:
    - metric_files (list): List of file paths.
    
    Returns:
    - pd.DataFrame: DataFrame containing MSE, MAE, RMSE for each sample.
    """
    data = []
    for file in metric_files:
        metrics = parse_metric_file(file)
        if metrics:  # Ensure that the file was parsed correctly
            data.append(metrics)
    df = pd.DataFrame(data)
    return df

def compute_statistics(df):
    """
    Computes summary statistics for each metric.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the metrics.
    
    Returns:
    - pd.DataFrame: DataFrame containing summary statistics.
    """
    stats = df.describe().loc[['mean', '50%', 'std', 'min', 'max', '25%', '75%']]
    stats.rename(index={'50%': 'median'}, inplace=True)
    return stats

def plot_histogram(df, metric, output_dir, save_plots=True, show_plots=False):
    """
    Plots a histogram with KDE for a specified metric.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the metrics.
    - metric (str): The metric to plot (e.g., 'MSE', 'MAE', 'RMSE').
    - output_dir (str): Directory to save the plot.
    - save_plots (bool): Whether to save the plot as an image.
    - show_plots (bool): Whether to display the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(df[metric], bins=30, kde=True, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {metric}')
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_plots:
        plot_path = os.path.join(output_dir, f'{metric}_distribution.png')
        plt.savefig(plot_path)
        print(f'Saved {metric} distribution plot to {plot_path}')
    
    if show_plots:
        plt.show()
    
    plt.close()

def plot_histogram_with_annotations(df, metric, output_dir, save_plots=True, show_plots=False):
    """
    Plots a histogram with KDE and annotations (mean and median) for a specified metric.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the metrics.
    - metric (str): The metric to plot.
    - output_dir (str): Directory to save the plot.
    - save_plots (bool): Whether to save the plot as an image.
    - show_plots (bool): Whether to display the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(df[metric], bins=30, kde=True, color='skyblue', edgecolor='black')
    
    # Compute statistics
    mean = df[metric].mean()
    median = df[metric].median()
    
    # Add vertical lines for mean and median
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean:.4f}')
    plt.axvline(median, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median:.4f}')
    
    plt.title(f'Distribution of {metric}')
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_plots:
        plot_path = os.path.join(output_dir, f'{metric}_distribution_with_annotations.png')
        plt.savefig(plot_path)
        print(f'Saved {metric} distribution plot with annotations to {plot_path}')
    
    if show_plots:
        plt.show()
    
    plt.close()

def plot_boxplot(df, metric, output_dir, save_plots=True, show_plots=False):
    """
    Plots a boxplot for a specified metric.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the metrics.
    - metric (str): The metric to plot.
    - output_dir (str): Directory to save the plot.
    - save_plots (bool): Whether to save the plot as an image.
    - show_plots (bool): Whether to display the plot.
    """
    plt.figure(figsize=(6, 8))
    sns.boxplot(y=df[metric], color='lightgreen')
    plt.title(f'Boxplot of {metric}')
    plt.ylabel(metric)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_plots:
        plot_path = os.path.join(output_dir, f'{metric}_boxplot.png')
        plt.savefig(plot_path)
        print(f'Saved {metric} boxplot to {plot_path}')
    
    if show_plots:
        plt.show()
    
    plt.close()

def plot_correlation_matrix(df, output_dir, save_plots=True, show_plots=False):
    """
    Plots a heatmap of the correlation matrix for the metrics.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the metrics.
    - output_dir (str): Directory to save the plot.
    - save_plots (bool): Whether to save the plot as an image.
    - show_plots (bool): Whether to display the plot.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Metrics')
    
    if save_plots:
        plot_path = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(plot_path)
        print(f'Saved correlation matrix plot to {plot_path}')
    
    if show_plots:
        plt.show()
    
    plt.close()

def main():
    """
    Main function to aggregate and visualize depth estimation metrics.
    """
    # Load configuration from 'inference.yaml'
    config_path = 'config/inference.yaml'
    config = load_config(config_path)  # Update the path if 'inference.yaml' is located elsewhere
    
    # Extract relevant parameters
    inference_params = config.get('inference_parameters', {})
    model_variant = inference_params.get('model_variant')
    models = inference_params.get('models', {})
    
    if not model_variant:
        print("No 'model_variant' specified in 'inference_parameters' of 'inference.yaml'.")
        return
    
    model_info = models.get(model_variant)
    if not model_info:
        print(f"Model variant '{model_variant}' not found in 'inference_parameters.models' of 'inference.yaml'.")
        return
    
    output_dir = model_info.get('output_dir')
    if not output_dir:
        print(f"'output_dir' not specified for model variant '{model_variant}' in 'inference.yaml'.")
        return
    
    metrics_dir = output_dir  # Metrics are saved directly in output_dir
    if not os.path.exists(metrics_dir):
        print(f"Metrics directory '{metrics_dir}' does not exist for model variant '{model_variant}'.")
        return
    
    # Find all metric files
    metric_files = find_metric_files(metrics_dir)
    print(f'Found {len(metric_files)} metric files in "{metrics_dir}".')
    
    if not metric_files:
        print(f'No metric files found in "{metrics_dir}" for model variant "{model_variant}".')
        return
    
    # Aggregate metrics
    df_metrics = aggregate_metrics(metric_files)
    print(f'Aggregated metrics for {len(df_metrics)} samples in "{model_variant}".')
    
    if df_metrics.empty:
        print(f'No metrics were extracted from the metric files in "{metrics_dir}".')
        return
    
    # Compute and display summary statistics
    stats = compute_statistics(df_metrics)
    print('\nSummary Statistics:')
    print(stats)
    
    """
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(metrics_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f'Created plots directory at "{plots_dir}".')
    
    # Plot histograms with annotations for each metric
    for metric in ['MSE', 'MAE', 'RMSE']:
        if metric in df_metrics.columns:
            plot_histogram_with_annotations(df_metrics, metric, plots_dir, save_plots=True, show_plots=False)
            #plot_boxplot(df_metrics, metric, plots_dir, save_plots=True, show_plots=False)
        else:
            print(f'Metric "{metric}" not found in the DataFrame for model variant "{model_variant}". Skipping plotting for this metric.')
    
    # Plot correlation matrix
    #plot_correlation_matrix(df_metrics, plots_dir, save_plots=True, show_plots=False)
    
    # Save aggregated metrics to a CSV file for future reference
    aggregated_csv_path = os.path.join(metrics_dir, 'aggregated_metrics.csv')
    df_metrics.to_csv(aggregated_csv_path, index=False)
    print(f'\nSaved aggregated metrics to "{aggregated_csv_path}" for model variant "{model_variant}".')

    """
    
if __name__ == "__main__":
    main()
