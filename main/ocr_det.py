import cv2
import numpy as np
import os
from barcode_det import erase_barcodes_from_image


def order_points(pts):
    """
    The `order_points` function takes a list of points and returns the points in a specific order based
    on their coordinates.

    :param pts: It seems like the code snippet you provided is for ordering points to form a rectangle.
    However, you have not provided the actual points (pts) that need to be ordered. Could you please
    provide the values of the points so that I can help you with ordering them to form a rectangle?
    :return: The `order_points` function returns a numpy array `rect` containing the four points of a
    rectangle. The points are ordered in a specific way based on the input points `pts`.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_and_transform_largest_table(image):
    """
    The function `extract_and_transform_largest_table` takes an image as input, identifies the largest
    table in the image, and warps the perspective to extract and transform the table.

    :param image: The code you provided seems to be related to extracting and transforming the largest
    table from an image using OpenCV. It includes functions for extracting and transforming the largest
    table as well as sorting contours based on different methods
    :return: The code provided includes two functions: `extract_and_transform_largest_table` and
    `sort_contours`. The `extract_and_transform_largest_table` function takes an image as input,
    processes it to extract the largest table-like structure, and returns the transformed image of the
    extracted table. If a suitable table is found in the image, it applies perspective transformation to
    extract and return the table. If no
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("Thresholded", thresh)
    # cv2.waitKey(0)

    ''' 
    # Show Drawing Contours
    image_with_contours = image.copy()    
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)
    cv2.imshow("Contours", image_with_contours)
    cv2.waitKey(0)
    '''

    best_cnt = None
    max_area = 0
    max_width = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = w / float(h)
        if aspect_ratio > 1.5 and area > 600:
            if area > max_area:
                max_area = area
                best_cnt = cnt
                max_width = w

    if best_cnt is not None:
        rect = cv2.minAreaRect(best_cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        rect_ordered = order_points(box)

        '''
        # Visual confirmation of the detected table outline
        image_copy = image.copy()
        cv2.polylines(image_copy, [np.int32(rect_ordered)], isClosed=True, color=(0, 255, 0), thickness=5)
        cv2.imshow("Detected Table Outline", image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        '''

        (tl, tr, br, bl) = rect_ordered
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        max_width = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        max_height = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect_ordered, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))

        # cv2.imshow("Warped Image", warped)
        # cv2.waitKey(0)
        return warped

    return image


def sort_contours(contours, method="top-to-bottom"):
    """
    The function `sort_contours` sorts a list of contours based on their bounding box coordinates in a
    specified direction.

    :param contours: Contours are the contours found in an image. In image processing, contours are
    continuous lines or curves that bound or cover the full boundary of an object in an image
    :param method: The `method` parameter in the `sort_contours` function specifies the direction in
    which the contours should be sorted. The possible values for the `method` parameter are:, defaults
    to top-to-bottom (optional)
    :return: The function `sort_contours` returns a tuple containing two elements: the sorted contours
    and their corresponding bounding boxes.
    """
    reverse = False
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    i = 0 if method == "left-to-right" or method == "right-to-left" else 1
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(
        *sorted(zip(contours, bounding_boxes), key=lambda b: b[1][i], reverse=reverse))
    return (contours, bounding_boxes)


# Load and preprocess the image


def process_input_file(file):
    """
    The function `process_input_file` reads an image file, erases barcodes, extracts the largest table,
    performs image processing operations, filters contours, sorts contours, groups contours into rows
    and columns, and saves each cell as an image file.

    :param file: The `file` parameter in the `process_input_file` function is the path to the input
    image file that you want to process. This function reads the image from the specified file, performs
    various image processing operations to extract and transform tables, and then extracts individual
    cells from the tables for further processing
    """
    img = cv2.imread(file)
    img = erase_barcodes_from_image(img)  # First erase barcodes
    # Then extract and transform the largest table
    img = extract_and_transform_largest_table(img)

    # Show Largest Table Being Extracted
    # cv2.imshow("largest image", img)
    # cv2.waitKey(0)

    # Convert to grayscale for further processing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, img_bin = cv2.threshold(
        img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin  # Inverting the image
    # Further image processing
    kernel_len = np.array(img).shape[1]//100
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))

    # Apply morphological operations
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)
    # cv2.imshow("vertical_lines", image_1)
    # cv2.waitKey(0)
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=1)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=4)
    # cv2.imshow("horizontal_lines", image_2)
    # cv2.waitKey(0)

    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.erode(~img_vh, kernel, iterations=4)
    # cv2.imshow("img_vh", img_vh)
    # cv2.waitKey(0)
    thresh, img_vh = cv2.threshold(
        img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find and sort contours for cell extraction
    contours, _ = cv2.findContours(
        img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    # Filter out noise cells
    min_area = 300
    min_width = 70
    min_height = 20
    # Ignore the Operation Details Cell
    max_width = 500
    filtered_contours_img = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > min_area and w > min_width and h > min_height:
            filtered_contours.append(cnt)
            cv2.rectangle(filtered_contours_img, (x, y),
                          (x + w, y + h), (0, 0, 255), 2)
    # cv2.imshow("Filtered Contours", filtered_contours_img)
    # cv2.waitKey(0)

    contours, bounding_boxes = sort_contours(
        filtered_contours, "top-to-bottom")

    # Group contours into rows and columns
    rows = []
    current_row = []
    prev_y = -1
    # Store center of the previous contour
    prev_center = None

    for contour, box in zip(contours, bounding_boxes):
        x, y, w, h = cv2.boundingRect(contour)
        # Calculate center of the current contour
        center = (x + w // 2, y + h // 2)
        # Adjust this threshold for distance between center and next row
        if prev_y == -1 or abs(center[1] - prev_center[1]) < h * 0.2:
            current_row.append((contour, box))
        else:
            rows.append(current_row)
            current_row = [(contour, box)]
        prev_y = y
        prev_center = center
    if current_row:  # Add the last row
        rows.append(current_row)

    # Calculate average cell size in column 3
    column_3_sizes = [cv2.boundingRect(cnt)[2] * cv2.boundingRect(cnt)[3]
                      for row in rows for cnt, box in row if 'column_3' in cv2.boundingRect(cnt)]
    avg_area_column_3 = sum(column_3_sizes) / \
        len(column_3_sizes) if column_3_sizes else 0

    # Set a threshold as a fraction of the average area to filter out small cells
    area_threshold_factor = 0.5  # Example: 50% of the average area
    area_threshold = avg_area_column_3 * area_threshold_factor

    # Crop and save each cell
    base_folder = "./tempTables"
    file_name_without_ext = os.path.splitext(os.path.basename(file))[0]
    individual_file_folder = os.path.join(base_folder, file_name_without_ext)
    os.makedirs(individual_file_folder, exist_ok=True)

    for row_idx, row in enumerate(rows[1:], start=2):
        column_contours, _ = sort_contours(
            [cnt[0] for cnt in row], "left-to-right")
        for cell_idx, (cnt, box) in enumerate(zip(column_contours, _)):
            x, y, w, h = box
            cell_area = w * h

            if cell_idx + 1 in [1, 2, 4]:  # Process columns 1, 2, and 4 as usual
                column_folder = os.path.join(
                    individual_file_folder, f"row_{row_idx}", f"column_{cell_idx+1}")
                os.makedirs(column_folder, exist_ok=True)
                cropped_cell = img[y:y+h, x:x+w]
                filename = os.path.join(
                    column_folder, f"cell_{cell_idx+1}.png")
                cv2.imwrite(filename, cropped_cell)
                print(f"Saved: {filename}")
            # For column 3, check against the dynamic area threshold
            elif cell_idx + 1 == 3 and cell_area >= area_threshold:
                column_folder = os.path.join(
                    individual_file_folder, f"row_{row_idx}", "column_3")
                os.makedirs(column_folder, exist_ok=True)
                cropped_cell = img[y:y+h, x:x+w]
                filename = os.path.join(
                    column_folder, f"cell_{cell_idx+1}.png")
                cv2.imwrite(filename, cropped_cell)
                print(f"Saved: {filename}")
    # /tempTables/production5000pg/rows


def take_input(image):
    """
    The `take_input` function processes an image file and prints "Executed".

    :param image: The `image` parameter in the `take_input` function is typically a file path or a
    reference to an image file that you want to process. It could be an image file in a specific format
    like JPEG, PNG, or any other supported image format. The function `process_input_file(image)`
    """
    process_input_file(image)
    print("Executed")


# Testing
# image = '500000294400_pages/page_3.png'
# take_input(image)
