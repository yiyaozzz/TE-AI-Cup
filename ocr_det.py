import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

# read your file
file = 'img4.png'
img = cv2.imread(file, 0)
img.shape

# thresholding the image to a binary image
thresh, img_bin = cv2.threshold(
    img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# inverting the image
img_bin = 255-img_bin
cv2.imwrite('cv_inverted.png', img_bin)
# Plotting the image to see the output
plotting = plt.imshow(img_bin, cmap='gray')
plt.show()

# countcol(width) of kernel as 100th of total width
kernel_len = np.array(img).shape[1]//100
# Defining a vertical kernel to detect all vertical lines of image
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
# Defining a horizontal kernel to detect all horizontal lines of image
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
# A kernel of 2x2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# Use vertical kernel to detect and save the vertical lines in a jpg
image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
cv2.imwrite("vertical.jpg", vertical_lines)
# Plot the generated image
plotting = plt.imshow(image_1, cmap='gray')
plt.show()

# Use horizontal kernel to detect and save the horizontal lines in a jpg
image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
cv2.imwrite("horizontal.jpg", horizontal_lines)
# Plot the generated image
plotting = plt.imshow(image_2, cmap='gray')
plt.show()

# Combine horizontal and vertical lines in a new third image, with both having same weight.
img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
# Eroding and thesholding the image
img_vh = cv2.erode(~img_vh, kernel, iterations=2)
thresh, img_vh = cv2.threshold(
    img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("img_vh.jpg", img_vh)
bitxor = cv2.bitwise_xor(img, img_vh)
bitnot = cv2.bitwise_not(bitxor)
# Plotting the generated image
plotting = plt.imshow(bitnot, cmap='gray')
plt.show()

# Detect contours for following box detection
contours, hierarchy = cv2.findContours(
    img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def sort_contours(cnts, method="top-to-bottom"):
    # Sorting helper function based on method
    reverse = True if method in ["right-to-left", "bottom-to-top"] else False
    i = 1 if method in ["top-to-bottom", "bottom-to-top"] else 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    cnts, boundingBoxes = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes


# Sort contours
contours, boundingBoxes = sort_contours(contours, "top-to-bottom")

# Group contours into rows based on their y-coordinate
rows = []
current_row = []
prev_y = -1
mean_height = np.mean([h for _, _, _, h in boundingBoxes])

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Adjust mean threshold as necessary
    if prev_y == -1 or abs(y - prev_y) < mean_height:
        current_row.append(cnt)
    else:
        rows.append(current_row)
        current_row = [cnt]
    prev_y = y
if current_row:  # Add the last row
    rows.append(current_row)


# Creating a list of heights for all detected boxes
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

# Get mean of heights
mean = np.mean(heights)

# Create list box to store all boxes in
box = []
# Get position (x,y), width and height for every contour and show the contour on image
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if (w < 1000 and h < 500):
        image = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        box.append([x, y, w, h])

plotting = plt.imshow(image, cmap='gray')
plt.show()

# Creating two lists to define row and column in which cell is located
row = []
column = []
j = 0

# Sorting the boxes to their respective row and column
for i in range(len(box)):

    if (i == 0):
        column.append(box[i])
        previous = box[i]

    else:
        if (box[i][1] <= previous[1]+mean/2):
            column.append(box[i])
            previous = box[i]

            if (i == len(box)-1):
                row.append(column)

        else:
            row.append(column)
            column = []
            previous = box[i]
            column.append(box[i])

print(column)
print(row)

# calculating maximum number of cells
countcol = 0
for i in range(len(row)):
    countcol = len(row[i])
    if countcol > countcol:
        countcol = countcol

# Retrieving the center of each column
center = [int(row[i][j][0]+row[i][j][2]/2)
          for j in range(len(row[i])) if row[0]]

center = np.array(center)
center.sort()
print(center)
# Regarding the distance to the columns center, the boxes are arranged in respective order

finalboxes = []
for i in range(len(row)):
    lis = []
    for k in range(countcol):
        lis.append([])
    for j in range(len(row[i])):
        diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
        minimum = min(diff)
        indexing = list(diff).index(minimum)
        lis[indexing].append(row[i][j])
    finalboxes.append(lis)


# from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
outer = []
for i in range(len(finalboxes)):
    for j in range(len(finalboxes[i])):
        inner = ''
        if (len(finalboxes[i][j]) == 0):
            outer.append(' ')
        else:
            for k in range(len(finalboxes[i][j])):
                y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], finalboxes[i][j][k][3]
                finalimg = bitnot[x:x+h, y:y+w]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                border = cv2.copyMakeBorder(
                    finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                resizing = cv2.resize(border, None, fx=2,
                                      fy=2, interpolation=cv2.INTER_CUBIC)
                dilation = cv2.dilate(resizing, kernel, iterations=1)
                erosion = cv2.erode(dilation, kernel, iterations=2)

                out = pytesseract.image_to_string(erosion)
                if (len(out) == 0):
                    out = pytesseract.image_to_string(
                        erosion, config='--psm 3')
                inner = inner + " " + out
            outer.append(inner)
output_folder = "./cropped_cells"  # Folder where cropped images will be saved
# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


def crop_and_save_cells(rows, img):
    # Skipping the first row if it's headers
    for row_idx, row in enumerate(rows[1:], start=1):
        for cell_idx, cnt in enumerate(row):
            x, y, w, h = cv2.boundingRect(cnt)
            cropped_cell = img[y:y+h, x:x+w]
            # Save the cropped image
            filename = f"cropped_row{row_idx}_cell{cell_idx}.png"
            cv2.imwrite(filename, cropped_cell)
            print(f"Saved: {filename}")


# Print the number of rows and columns excluding the first row
crop_and_save_cells(rows, img)


# For each row, print the number of columns (cells)
for i, row in enumerate(finalboxes[1:], start=2):  # Start from the second row
    print(f"Row {i} has {len(row)} columns")
# Creating a dataframe of the generated OCR list
arr = np.array(outer)
dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
print(dataframe)
data = dataframe.style.set_properties(align="left")
# Converting it in a excel-file
data.to_excel("/Users/marius/Desktop/output.xlsx")
