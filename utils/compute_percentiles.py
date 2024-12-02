import numpy as np
from tools import compute_global_percentiles
import os

def main():
    """
    Computes global percentiles (P1 and P99) for the depth maps and saves them to a file.
    """
    # Define the path to the depth maps directory
    depth_maps_folder = os.path.join('data', 'cedarbay', 'depth_maps')

    # Define the output path for percentiles
    percentiles_output_path = os.path.join('utils', 'percentiles.npz')

    # Compute global percentiles
    P1, P99 = compute_global_percentiles(depth_maps_folder, lower=1.0, upper=99.0)
    print(f"Computed Global Percentiles - P1: {P1}, P99: {P99}")

    # Save the percentiles to a .npz file
    np.savez(percentiles_output_path, P1=P1, P99=P99)
    print(f"Saved percentiles to {percentiles_output_path}")

if __name__ == "__main__":
    main()