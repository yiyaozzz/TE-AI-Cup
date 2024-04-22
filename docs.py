import os
from PIL import Image


def pad_image_to_size(image, target_width=640, target_height=360):
    current_width, current_height = image.size
    if current_width == target_width and current_height == target_height:
        return image
    else:
        new_image = Image.new(
            "RGB", (target_width, target_height), (255, 255, 255))
        x_offset = (target_width - current_width) // 2
        y_offset = (target_height - current_height) // 2
        new_image.paste(image, (x_offset, y_offset))
        return new_image


def process_directory_structure(base_path, save_path):
    i = 500
    allowed_columns = ['column_1', 'column_2',
                       'column_3']
    for dirpath, dirnames, filenames in os.walk(base_path):
        if os.path.basename(dirpath) in allowed_columns:
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(dirpath, filename)
                    with Image.open(image_path) as img:
                        padded_image = pad_image_to_size(img)

                    base_filename, file_extension = os.path.splitext(filename)
                    save_image_path = os.path.join(
                        save_path, f"{base_filename}_{i}{file_extension}")

                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    padded_image.save(save_image_path)
                    print(f"Processed and saved {save_image_path}")
                    i += 1


# Example usage
base_path = 'finalOutput'
save_path = 'processedOutput'
process_directory_structure(base_path, save_path)
