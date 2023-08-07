import os
import cv2
import json

# Define your project directory as an absolute path
project_directory = r'C:\Users\\cardo\\Documents\\edrich\\project_optel'

# Define raw data directory and activity folders
raw_data_directory = os.path.join(project_directory, 'data', 'raw-data')
activity_folders = ['activity1', 'activity2', 'activity3', 'activity4', 'activity5']

# Define processed data directory
processed_data_directory = os.path.join(project_directory, 'data', 'processed_data')

# Define correction factors for body parts (example values)
correction_factors = {
    "shoulder": {"x": 0, "y": -10},
    "elbow": {"x": 0, "y": 0},
    "wrist": {"x": 0, "y": 5},
    "hip": {"x": 0, "y": 10},
    "knee": {"x": 0, "y": 15},
    "ankle": {"x": 0, "y": 10}
}

# Define callback function for mouse click events
def annotate_image(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Correct keypoints based on body part
        body_part = param["body_part"]
        correction = correction_factors.get(body_part, {"x": 0, "y": 0})
        corrected_x = x + correction["x"]
        corrected_y = y + correction["y"]

        # Add the corrected keypoint to the list
        param["annotations"].append({"x": corrected_x, "y": corrected_y, "body_part": body_part})

        # Draw a circle to visualize the corrected keypoint on the image
        cv2.circle(image, (corrected_x, corrected_y), 3, (0, 255, 0), -1)
        cv2.imshow('Image', image)

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

        # Create a window and set the mouse callback function
        cv2.imshow('Image', image)
        annotations = []  # Initialize annotations list for each image
        cv2.setMouseCallback('Image', annotate_image, {"body_part": "shoulder", "annotations": annotations})

        # Wait for user to annotate keypoints and press 'ESC' to move to the next image
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # 'ESC' key
                break
            elif key == ord('s'):  # Change body part to shoulder
                cv2.setMouseCallback('Image', annotate_image, {"body_part": "shoulder", "annotations": annotations})
            elif key == ord('e'):  # Change body part to elbow
                cv2.setMouseCallback('Image', annotate_image, {"body_part": "elbow", "annotations": annotations})
            elif key == ord('w'):  # Change body part to wrist
                cv2.setMouseCallback('Image', annotate_image, {"body_part": "wrist", "annotations": annotations})
            elif key == ord('h'):  # Change body part to head
                cv2.setMouseCallback('Image', annotate_image, {"body_part": "head", "annotations": annotations})
            elif key == ord('i'):  # Change body part to hip
                cv2.setMouseCallback('Image', annotate_image, {"body_part": "hip", "annotations": annotations})
            elif key == ord('k'):  # Change body part to knee
                cv2.setMouseCallback('Image', annotate_image, {"body_part": "knee", "annotations": annotations})
            elif key == ord('a'):  # Change body part to ankle
                cv2.setMouseCallback('Image', annotate_image, {"body_part": "ankle", "annotations": annotations})

        # Correct keypoints based on correction factors
        for keypoint in annotations:
            body_part = keypoint["body_part"]
            correction = correction_factors.get(body_part, {"x": 0, "y": 0})
            keypoint["x"] += correction["x"]
            keypoint["y"] += correction["y"]

        # Save the corrected annotations to a JSON file
        annotation_data = {
            "filename": filename,
            "activity": activity_folder,
            "annotations": annotations
        }
        processed_annotation_path = os.path.join(processed_data_directory, activity_folder, filename[:-4] + "_annotated.json")
        os.makedirs(os.path.dirname(processed_annotation_path), exist_ok=True)
        with open(processed_annotation_path, 'w') as f:
            json.dump(annotation_data, f, indent=4)

        # Close the image window
        cv2.destroyAllWindows()

print("Annotations corrected and saved to processed data directory.")
