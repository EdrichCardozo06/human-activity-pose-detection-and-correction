import os
import cv2
import json

# Define your project directory as an absolute path
project_directory = r'C:\\Users\\cardo\\Documents\\edrich\\project_optel'

# Define raw data directory and activity folders
raw_data_directory = os.path.join(project_directory, 'data', 'raw-data')
activity_folders = ['activity1', 'activity2', 'activity3', 'activity4', 'activity5']

# Define processed data directory
processed_data_directory = os.path.join(project_directory, 'data', 'processed_data')

# Define correction factors for keypoints (example values)
correction_factors = {
    "shoulder": {"x": 0, "y": -10},
    "elbow": {"x": 0, "y": 0},
    "wrist": {"x": 0, "y": 10},
    "hip": {"x": 0, "y": 20},
    "knee": {"x": 0, "y": 10}
    # Add more keypoints as needed
}

# Load image filenames from activity folders
for activity_folder in activity_folders:
    activity_path = os.path.join(raw_data_directory, activity_folder)
    
    if not os.path.exists(activity_path):
        print(f"Warning: Activity folder '{activity_folder}' not found.")
        continue
    
    image_filenames = os.listdir(activity_path)

    for filename in image_filenames:
        # Load image
        image_path = os.path.join(activity_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Failed to load image '{filename}' in '{activity_folder}'")
            continue

        # Correct keypoints based on correction factors
        keypoints = []
        for keypoint, correction in correction_factors.items():
            x = image.shape[1] // 2  # Example: x-coordinate at the center of the image
            y = image.shape[0] // 2  # Example: y-coordinate at the center of the image
            corrected_x = x + correction["x"]
            corrected_y = y + correction["y"]
            keypoints.append({"x": corrected_x, "y": corrected_y, "body_part": keypoint})

        # Save the corrected keypoints to a JSON file
        annotation_data = {
            "filename": filename,
            "activity": activity_folder,
            "keypoints": keypoints
        }
        processed_annotation_path = os.path.join(processed_data_directory, activity_folder, filename[:-4] + "_keypoints.json")
        os.makedirs(os.path.dirname(processed_annotation_path), exist_ok=True)
        with open(processed_annotation_path, 'w') as f:
            json.dump(annotation_data, f, indent=4)

print("Keypoints corrected and saved to processed data directory.")
