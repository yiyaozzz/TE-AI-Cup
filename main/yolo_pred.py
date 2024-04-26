from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import os
import time


model = YOLO('model/best4.pt')


def create_directory_structure(base_dir, image_path):
    os.makedirs(base_dir, exist_ok=True)
    parts = image_path.split('/')
    page_index = next(i for i, part in enumerate(parts) if 'page_' in part)
    path_to_create = os.path.join(base_dir, *parts[page_index:-1])

    os.makedirs(path_to_create, exist_ok=True)
    return path_to_create


def group_detections_by_rows(detections, vertical_threshold=10):
    rows = []
    for det in detections:
        _, y1, _, _, _ = det
        found_row = False
        for row in rows:
            if any(abs(y1 - existing_y1) <= vertical_threshold for _, existing_y1, _, _, _ in row):
                row.append(det)
                found_row = True
                break
        if not found_row:
            rows.append([det])
    return rows


def track_object(directory_path, base_output_dir='finalOutput'):
    if not os.path.isdir(directory_path):
        print(f"Error: The provided path {directory_path} is not a directory.")
        return

    supported_image_types = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(
        directory_path) if f.endswith(supported_image_types)]

    if not image_files:
        print(f"No images found in {directory_path}.")
        return

    image_files.sort()  # Sort the image files to maintain order

    for image_filename in image_files:
        image_path = os.path.join(directory_path, image_filename)
        output_dir = create_directory_structure(base_output_dir, image_path)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Image could not be read from path {image_path}.")
            continue

        results = model.predict(img, conf=0.2, iou=0.1)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls)]
                detections.append((x1, y1, x2, y2, label))

        # Group detections into rows
        rows = group_detections_by_rows(detections)

        # Within each row, sort by label ('Words' first) and then by x1
        for row in rows:
            row.sort(key=lambda x: (x[4] != 'Words', x[0]))

        # Save each detection, organized by row and labeled sequentially
        file_index = 1
        for row in rows:
            for x1, y1, x2, y2, label in row:
                cropped_img = img[y1:y2, x1:x2]
                filename = f"{file_index}_{label}.png"
                cv2.imwrite(os.path.join(output_dir, filename), cropped_img)
                file_index += 1


# Example of how to call the function
# directory_path = 'tempTables/page_9/row_3/column_4'
# track_object(directory_path)
