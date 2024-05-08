import cv2
import os


def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(
            f"Image not found or unable to load the image at the path: {image_path}")

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blurring to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to binarize the image
    adaptive_threshold = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)

    # Define a kernel size for morphological operations to connect letters into words
    # Wider kernel to connect letters horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))

    # Perform morphological operations to close gaps between letters, forming words
    morph = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_CLOSE, kernel)

    return morph, img  # Return both processed and original image for further use


def find_and_draw_contours(processed_image, original_image, save_directory):
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(
        processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw bounding boxes
    image_with_boxes = original_image.copy()

    # Check if save directory exists, if not, create it
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Iterate through each contour found and extract the word
    for index, contour in enumerate(contours):
        # Get the bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        # Check if the contour is large enough to be considered a word
        if w > 20 and h > 10:  # Adjust these thresholds based on your specific image scale
            # Draw a green rectangle to visualize the bounding rect
            cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image_with_boxes


def main():
    # Example usage
    try:
        # Replace with your actual image path
        image_path = 'nextjs/works/finalOutput_259e9b1d-6f19-43e8-8316-660dc698c88d.pdf/page_14/row_2/column_4/19_Words.png'
        processed_image, original_image = preprocess_image(image_path)
        save_directory = 'extracted_words'
        image_with_boxes = find_and_draw_contours(
            processed_image, original_image, save_directory)

        # Display the image with bounding boxes
        cv2.imshow("Words with Bounding Boxes", image_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    main()
