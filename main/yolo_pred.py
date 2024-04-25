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


def track_object(directory_path, base_output_dir='finalOutput'):
    # time.sleep(4)

    # Ensure the path is a directory
    if not os.path.isdir(directory_path):
        print(f"Error: The provided path {directory_path} is not a directory.")
        return

    # Find image files in the directory
    supported_image_types = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(
        directory_path) if f.endswith(supported_image_types)]

    if not image_files:
        print(f"No images found in {directory_path}.")
        return

    for image_filename in image_files:
        image_path = os.path.join(directory_path, image_filename)
        output_dir = create_directory_structure(base_output_dir, image_path)
        img = cv2.imread(image_path)
        if img is None:
            print(
                f"Error: Image could not be read from path {image_path}. Check the path and file permissions.")
            continue

        results = model.predict(img, conf=0.2, iou=0.1)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls)]
                detections.append((x1, y1, x2, y2, label))

        # Function to determine rows based on vertical overlap
        def get_row(y, rows):
            for idx, row in enumerate(rows):
                for _, row_y1, _, row_y2, _ in row:
                    if abs(row_y1 - y) <= 10 or abs(row_y2 - y) <= 10:
                        return idx
            return -1

        # Group detections into rows
        rows = []
        for det in sorted(detections, key=lambda x: (x[1], x[0])):
            idx = get_row(det[1], rows)
            if idx == -1:
                rows.append([det])
            else:
                rows[idx].append(det)

        # Save each detection, organized by row
        file_index = 1
        for row in rows:
            for x1, y1, x2, y2, label in sorted(row, key=lambda x: x[0]):
                cropped_img = img[y1:y2, x1:x2]
                filename = f"{file_index}_{label}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), cropped_img)
                file_index += 1


# Example of how to call the function
# directory_path = 'finalOutput/page_6/row_2/column_2'
# track_object(directory_path)
