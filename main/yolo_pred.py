from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import os

model = YOLO('model/best2.pt')


image_path = 'tempTables/page_9/row_3/column_4/cell_4.png'
img = cv2.imread(image_path)


results = model.predict(img, conf=0.2, iou=0.5)

detections = []
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls)]
        detections.append((x1, y1, x2, y2, label))


def get_row(y, rows, threshold=10):
    for idx, row in enumerate(rows):
        for _, row_y1, _, row_y2, _ in row:
            if abs(row_y1 - y) <= threshold or abs(row_y2 - y) <= threshold:
                return idx
    return -1


rows = []

for det in sorted(detections, key=lambda x: (x[1], x[0])):
    idx = get_row(det[1], rows)
    if idx == -1:
        rows.append([det])
    else:
        rows[idx].append(det)


output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Save each detection, organized by row
file_index = 1
for row in rows:
    for x1, y1, x2, y2, label in sorted(row, key=lambda x: x[0]):
        cropped_img = img[y1:y2, x1:x2]
        filename = f"{file_index}_{label}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), cropped_img)
        file_index += 1

annotator = Annotator(img)
for row in rows:
    for x1, y1, x2, y2, label in row:
        annotator.box_label([x1, y1, x2, y2], label)
annotated_img = annotator.result()
cv2.imshow('YOLO V8 Detection', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
