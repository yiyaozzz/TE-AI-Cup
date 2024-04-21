import os
import shutil
import re
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('model/best2.pt')


def get_page_from_path(image_path):
    path_parts = image_path.split(os.sep)
    page_pattern = re.compile(r'^page_\d+$')
    for part in path_parts:
        if page_pattern.match(part):
            return part
    return None


# Set up output directory
output_dir = "finalOutput"
os.makedirs(output_dir, exist_ok=True)

track_counter = 3


def track_object(source_path):
    global track_counter

    page_dir = get_page_from_path(source_path)
    if not page_dir:
        print("Page directory not found in the path.")
        return None

    page_output_dir = os.path.join(output_dir, page_dir)
    os.makedirs(page_output_dir, exist_ok=True)

    track_counter += 1  # Ensure this logic is correct for your use case

    # Default output path for YOLOv5 is typically 'runs/detect/exp'
    # We track objects using YOLO
    results = model.track(
        source=source_path,
        tracker="bytetrack.yaml",
        conf=0.2,
        show=False,
        save=True,
        save_crop=True,
        save_conf=False,
        iou=0.5
    )

    # Find the last created directory within 'runs/detect' which contains the results
    # Adjust if your path is different
    default_save_path = os.path.join('runs', 'detect')
    try:
        latest_run_dir = max([os.path.join(default_save_path, d)
                             for d in os.listdir(default_save_path)], key=os.path.getmtime)
        # Move all files from the latest run directory to your specified directory
        for filename in os.listdir(latest_run_dir):
            source_file = os.path.join(latest_run_dir, filename)
            destination_file = os.path.join(
                page_output_dir, f"cell_{track_counter}_{filename}")
            shutil.move(source_file, destination_file)
        shutil.rmtree(latest_run_dir)  # Clean up the original directory
    except Exception as e:
        print(f"Error in handling YOLO output files: {e}")

    # Reset counter logic if necessary
    if track_counter == 4:
        track_counter = 3

    return results


# Example usage
track_object("path/to/your/image/file.jpg")
