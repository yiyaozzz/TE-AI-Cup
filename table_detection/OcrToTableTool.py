import cv2
import numpy as np
import subprocess

class OcrToTableTool:

    def __init__(self, image, original_image):
        self.thresholded_image = image
        self.original_image = original_image

    def execute(self):
        self.dilate_image()
        self.find_contours()
        self.convert_contours_to_bounding_boxes()
        self.mean_height = self.get_mean_height_of_bounding_boxes()
        self.sort_bounding_boxes_by_y_coordinate()
        self.club_all_bounding_boxes_by_similar_y_coordinates_into_rows()
        self.identify_and_mask_columns()
        self.crop_each_bounding_box_and_ocr()
        self.generate_csv_file()
        
    # New method to identify and mask specified columns
    def identify_and_mask_columns(self):
        # Assuming columns are sorted and filtered
        self.sort_and_filter_all_rows_by_x_coordinate()
        
        # Mask columns 3 and 6 by drawing a white rectangle over them
        for row in self.rows:
            for i, (x, y, w, h) in enumerate(row):
                # Check if it's the 3rd or 6th column index after sorting and filtering
                if i == 2 or i == 5:  # Adjust these indices based on actual column numbers after filtering
                    cv2.rectangle(self.original_image, (x, y), (x+w, y+h), (255, 255, 255), -1)  # -1 fills the rectangle

    def sort_and_filter_all_rows_by_x_coordinate(self):
        for row in self.rows:
            row.sort(key=lambda x: x[0])
            # Eliminate columns 3 and 6 from each row
            if len(row) > 6:  # Ensure there are at least 7 columns
                del row[5]  # Delete column 6 first to not affect the index of column 3
                del row[2]  # Now delete column 3

    def threshold_image(self):
        return cv2.threshold(self.grey_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def convert_image_to_grayscale(self):
        return cv2.cvtColor(self.image, self.dilated_image)

    def dilate_image(self):
        kernel_to_remove_gaps_between_words = np.array([
                [1,1,1,1,1,1,1,1,1,1],
               [1,1,1,1,1,1,1,1,1,1]
        ])
        self.dilated_image = cv2.dilate(self.thresholded_image, kernel_to_remove_gaps_between_words, iterations=5)
        simple_kernel = np.ones((5,5), np.uint8)
        self.dilated_image = cv2.dilate(self.dilated_image, simple_kernel, iterations=2)
    
    def find_contours(self):
        result = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = result[0]
        self.image_with_contours_drawn = self.original_image.copy()
        cv2.drawContours(self.image_with_contours_drawn, self.contours, -1, (0, 255, 0), 3)
    
    def approximate_contours(self):
        self.approximated_contours = []
        for contour in self.contours:
            approx = cv2.approxPolyDP(contour, 3, True)
            self.approximated_contours.append(approx)

    def draw_contours(self):
        self.image_with_contours = self.original_image.copy()
        cv2.drawContours(self.image_with_contours, self.approximated_contours, -1, (0, 255, 0), 5)

    def convert_contours_to_bounding_boxes(self):
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.original_image.copy()
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            self.bounding_boxes.append((x, y, w, h))
            self.image_with_all_bounding_boxes = cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 5)

    def get_mean_height_of_bounding_boxes(self):
        heights = []
        for bounding_box in self.bounding_boxes:
            x, y, w, h = bounding_box
            heights.append(h)
        return np.mean(heights)

    def sort_bounding_boxes_by_y_coordinate(self):
        self.bounding_boxes = sorted(self.bounding_boxes, key=lambda x: x[1])

    def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self):
        self.rows = []
        half_of_mean_height = self.mean_height / 2
        current_row = [ self.bounding_boxes[0] ]
        for bounding_box in self.bounding_boxes[1:]:
            current_bounding_box_y = bounding_box[1]
            previous_bounding_box_y = current_row[-1][1]
            distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)
            if distance_between_bounding_boxes <= half_of_mean_height:
                current_row.append(bounding_box)
            else:
                self.rows.append(current_row)
                current_row = [ bounding_box ]
        self.rows.append(current_row)

    def sort_all_rows_by_x_coordinate(self):
        for row in self.rows:
            row.sort(key=lambda x: x[0])
   

    def crop_each_bounding_box_and_ocr(self):
        self.table = []
        image_number = 0  # Define image_number here
        for row in self.rows:
            current_row = []
            for bounding_box in row:
                x, y, w, h = bounding_box
                y = y - 5  # Small adjustment
                cropped_image = self.original_image[y:y+h, x:x+w]
                image_slice_path = "./ocr_slices/image_" + str(image_number) + ".png"
                cv2.imwrite(image_slice_path, cropped_image)
                results_from_ocr = self.get_result_from_tesseract(image_slice_path)
                current_row.append(results_from_ocr)
                image_number += 1  # Increment image_number for the next slice
            self.table.append(current_row)

    def show_updated_table(self):
        # Create a blank image to put text on
        image_height = 20 * len(self.table)  # Adjust based on the number of rows and desired font size
        image_width = 800  # Arbitrary width, adjust as needed
        blank_image = np.zeros((image_height, image_width, 3), np.uint8)
        blank_image.fill(255)  # Make the background white

        # Set font parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)
        line_type = 2
        start_y = 15  # Starting Y position

        # Loop over table rows and put them on the image
        for row in self.table:
            text = " | ".join(row)  # Concatenate the row's text with a separator
            cv2.putText(blank_image, text, (10, start_y), font, font_scale, font_color, line_type)
            start_y += 20  # Move to the next row position

        # Display the image
        cv2.imshow("Updated Table", blank_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_result_from_tesseract(self, image_path):
        output = subprocess.getoutput('tesseract ' + image_path + ' stdout -l eng --oem 3 --psm 7 --dpi 72 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789().calmg* "')
        output = output.strip()
        return output

    def generate_csv_file(self):
        with open("output.csv", "w") as f:
            for row in self.table:
                f.write(",".join(row) + "\n")

    def store_process_image(self, file_name, image):
        path = "./process_images/ocr_table_tool/" + file_name
        cv2.imwrite(path, image)