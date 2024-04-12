import cv2
import numpy as np
from scipy.stats import mode
from sklearn.cluster import DBSCAN


def adaptive_dilation(image, initial_thresh):
    contours, _ = cv2.findContours(
        initial_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return initial_thresh

    heights = [cv2.boundingRect(c)[3] for c in contours]
    if heights:
        # find the most common height
        common_height = mode(heights)[0][0]
        # Scale the kernel based on common height
        kernel_height = max(2, int(common_height * 0.7))
    else:
        kernel_height = 5  # Default if no valid mode found

    # Slightly wider kernel
    kernel = np.ones(
        (kernel_height, max(2, int(kernel_height * 0.8))), np.uint8)
    dilated = cv2.dilate(initial_thresh, kernel, iterations=1)
    cv2.imshow('Result', dilated)
    cv2.waitKey(0)
    return dilated


def draw_bounding_boxes(image_path, eps, min_samples):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = adaptive_dilation(image, binary)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes.sort(key=lambda x: (x[1], x[0]))

    centroids = np.array([(x + w/2, y + h/2)
                         for (x, y, w, h) in bounding_boxes])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
    labels = clustering.labels_

    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue
        class_member_mask = (labels == label)
        rect_coords = [bounding_boxes[i]
                       for i in range(len(bounding_boxes)) if class_member_mask[i]]
        if rect_coords:
            x_min, y_min = np.min(rect_coords, axis=0)[0:2]
            x_max, y_max = np.max(rect_coords, axis=0)[
                0:2] + np.max(rect_coords, axis=0)[2:4]
            cv2.rectangle(image, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
draw_bounding_boxes('img9.png', eps=12, min_samples=1)
