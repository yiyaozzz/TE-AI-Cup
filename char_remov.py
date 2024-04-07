import cv2
import numpy as np

# Read the image
image_path = "img9.png"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform Otsu thresholding to get a binary image
_, binary_image = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform distance transform on the binary image
distance_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

# Set a threshold to create a binary image from the distance transform
_, binary_distance_transform = cv2.threshold(
    distance_transform, 0.7 * distance_transform.max(), 255, 0)

# Convert the binary distance transform to an 8-bit unsigned integer (0-255)
binary_distance_transform_uint8 = np.uint8(binary_distance_transform)

# Convert the binary distance transform to a 3-channel image
binary_distance_transform_3ch = cv2.merge(
    [binary_distance_transform_uint8] * 3)

# Convert the binary distance transform to the correct datatype for watershed
markers = np.int32(binary_distance_transform_uint8)

# Perform watershed on the original binary image using the distance transform as the marker
cv2.watershed(image, markers)

# Create a mask using the watershed markers
mask = np.uint8(markers)

# Find contours in the mask
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours from left to right
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

# Crop the words based on contours and save them
cropped_images_paths = []

for idx, contour in enumerate(contours):
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    # Crop the image using the bounding box
    cropped_image = image[y:y+h, x:x+w]
    # Save the cropped image
    cropped_image_path = f"cropped_word_{idx}.png"
    cv2.imwrite(cropped_image_path, cropped_image)
    cropped_images_paths.append(cropped_image_path)

image_with_boxes = image.copy()

# Draw the bounding box for each contour
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Since we cannot create pop-up windows in this environment,
# we will save the image with the drawn bounding boxes and provide it as a download.

# Save the image with the bounding boxes
boxed_image_path = "image_with_boxes.png"
cv2.imwrite(boxed_image_path, image_with_boxes)

boxed_image_path

cropped_images_paths


#  import cv2
# import numpy as np

# # Read the image
# image_path = "img8.png"
# image = cv2.imread(image_path)

# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Threshold the image to get a binary image
# _, binary_image = cv2.threshold(
#     gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# # Define a custom kernel for the morphological operation to connect letters
# custom_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 5))

# # Use morphological close operation to connect components/letters in the image
# connected = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, custom_kernel)
# morphed_image_path = "morphed_image.png"
# cv2.imwrite(morphed_image_path, connected)
# # Find contours in the connected image
# contours, _ = cv2.findContours(
#     connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Sort the contours from left to right
# contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

# # Crop the words based on contours and save them
# cropped_images_paths = []

# for idx, contour in enumerate(contours):
#     # Get the bounding box of the contour
#     x, y, w, h = cv2.boundingRect(contour)
#     # Crop the image using the bounding box
#     cropped_image = image[y:y+h, x:x+w]
#     # Save the cropped image
#     cropped_image_path = f"cropped_word_{idx}.png"
#     cv2.imwrite(cropped_image_path, cropped_image)
#     cropped_images_paths.append(cropped_image_path)

# image_with_boxes = image.copy()

# # Draw the bounding box for each contour
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

# # Since we cannot create pop-up windows in this environment,
# # we will save the image with the drawn bounding boxes and provide it as a download.

# # Save the image with the bounding boxes
# boxed_image_path = "image_with_boxes.png"
# cv2.imwrite(boxed_image_path, image_with_boxes)

# boxed_image_path

# cropped_images_paths
