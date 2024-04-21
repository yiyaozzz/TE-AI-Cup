import os
import re
from ultralytics import YOLO


model = YOLO('model/best2.pt')  # Adjust path as necessary


def get_page_from_path(image_path):
    path_parts = image_path.split(os.sep)
    page_pattern = re.compile(r'^page_\d+$')
    for part in path_parts:
        if page_pattern.match(part):
            return part
    return None


# Global variables
output_dir = "finalOutput"
os.makedirs(output_dir, exist_ok=True)

track_counter = 0


def track_object(source_path):
    global track_counter

    page_dir = get_page_from_path(source_path)
    if not page_dir:
        print("Page directory not found in the path.")
        return None

    page_output_dir = os.path.join(output_dir, page_dir)
    os.makedirs(page_output_dir, exist_ok=True)

    # Increment the track counter for this session
    track_counter += 1

    # Define the output filename within the appropriate directory
    output_filename = os.path.join(
        page_output_dir, f"cell_{track_counter}.png")

    # Run the tracking model
    results = model.track(
        source=source_path,
        tracker="bytetrack.yaml",
        conf=0.2,
        show=False,
        save=False,
        save_crop=True,
        save_conf=False,
        iou=0.5,
        # This should reflect in the saving behavior of the model
        name=f'track_{track_counter}'
    )

    # Additional handling if specific actions are needed with results
    return results


# track_object("500000294405_pages/page_2.png")
