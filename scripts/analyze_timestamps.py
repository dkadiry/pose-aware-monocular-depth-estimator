import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.tools import collect_list_of_image_timestamps
import pandas as pd

def main():
    path = "data\cedarbay\images"
    depth_map_path = "data\cedarbay\depth_maps"

    
    pose_csv_path = os.path.join("data", "cedarbay", "computed_FFC_relative_pose_data.csv")

    poses_df = pd.read_csv(pose_csv_path)
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

        missing_in_image = dmap_set - im_timestamp_set
        missing_in_dmap = im_timestamp_set - dmap_set
        missing_in_pose = im_timestamp_set - pose_timestamp_set
        missing_in_posed = dmap_set - pose_timestamp_set 

        if missing_in_dmap or missing_in_image or missing_in_pose or missing_in_posed:
            print(f'Missing in Image timestamp but present in Dmap timestamps: {missing_in_image}')
            print(f"Missing in Dmap timestamps, but present in Image timestamps: {missing_in_dmap}")
            print(f"Missing in Pose CSV file but present in Imgaes Folder: {missing_in_pose}")
            print(f"Missing in Pose CSV but present in Depth Map folder: {missing_in_posed}")

        else:
            print(f"Timestamps are aligned between datasets")

    else:
        print("Sets have either different or multiple data types")

if __name__ == "__main__":
    main()