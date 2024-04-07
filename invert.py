import cv2
import os


def invert_images_in_directory(source_directory, target_directory=None):
    """
    Inverts the colors of all images in the source_directory and saves them to the target_directory.

    :param source_directory: Path to the directory containing images to invert.
    :param target_directory: Path to the directory where inverted images will be saved. If None, saves in source_directory.
    """
    if target_directory is None:
        target_directory = source_directory

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for filename in os.listdir(source_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct full file path
            source_path = os.path.join(source_directory, filename)
            target_path = os.path.join(target_directory, filename)

            # Read the image
            img = cv2.imread(source_path)
            if img is None:
                print(f"Failed to load image: {source_path}")
                continue

            # Invert the image
            inverted_img = 255 - img

            # Save the inverted image
            cv2.imwrite(target_path, inverted_img)
            print(f"Inverted image saved to: {target_path}")


# Example usage
source_directory = 'datasets/val/1'
target_directory = 'datasets/1'  # Leave as None to overwrite source images
invert_images_in_directory(source_directory, target_directory)
