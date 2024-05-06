import cv2
from PIL import Image
import os
from ultralytics import YOLO


model = YOLO('model/best_Tally.pt')


def predict_and_show_labels(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image at {image_path}")
        return

    # Convert BGR (OpenCV format) to RGB (PIL format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Perform prediction using the model
    results = model.predict(img_pil)

    # Since results is a list, we take the first item if available
    if results:
        result = results[0]
        print("Detected labels:")
        if hasattr(result, 'boxes'):
            for box in result.boxes:
                # Ensure the class ID is an integer
                # Use .item() to extract a Python number from a tensor
                class_id = int(box.cls.item())
                # Retrieve the label using class index
                label = result.names[class_id]
                # Print label and confidence score
                print(f"{label}")

                return label


# image_path = "finalOutput/page_10/row_3/column_4/2_Words-and-tallys.png"
# predict_and_show_labels(image_path)
