import cv2
import numpy as np
import os


img = cv2.imread('img10.png')


def thresholding(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.GaussianBlur(thresh, (7, 7), 0)
    ret, thresh = cv2.threshold(thresh, 70, 255, cv2.THRESH_BINARY)
    # 100, 255
    return thresh


thresh_img = thresholding(img)
# thresh_img_no_hyphens = remove_hyphens(thresh_img)

cv2.imshow("thresh_img", thresh_img)

linesArray = []
kernelRows = np.ones((7, 8), np.uint8)

dilated = cv2.dilate(thresh_img, kernelRows, iterations=6)
cv2.imshow("dilated", dilated)

(contoursRows, heirarchy) = cv2.findContours(
    dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

linesArray = []
for row in contoursRows:
    area = cv2.contourArea(row)
    if area > 500:
        x, y, w, h = cv2.boundingRect(row)
        linesArray.append([x, y, w, h])

sortedLinesArray = sorted(linesArray, key=lambda line: line[1])

# 4.9 val
kernelWords = np.ones((int(1), int(11)), np.uint8)
dilateWordsImg = cv2.dilate(thresh_img, kernelWords, iterations=1)
cv2.imshow("dilate Words Img", dilateWordsImg)
img_with_bounds = img.copy()

output_dir = 'words'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


wordCount = 0
for line in sortedLinesArray:
    x, y, w, h = line
    roi_line = dilateWordsImg[y:y+h, x:x+w]
    (contoursWords, hierarchy) = cv2.findContours(
        roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sortedWords = sorted([cv2.boundingRect(word)
                         for word in contoursWords], key=lambda word: word[1])

    for word in sortedWords:
        x1, y1, w1, h1 = word
        roiWord = img[y+y1:y+y1+h1, x+x1:x+x1+w1]
        word_image_path = os.path.join(output_dir, f'word_{wordCount}.png')
        cv2.rectangle(img_with_bounds, (x+x1, y+y1),
                      (x+x1+w1, y+y1+h1), (0, 255, 0), 2)
        cv2.imwrite(word_image_path, roiWord)
        wordCount += 1

# Display the image with bounding boxes
cv2.imshow("Show the words", img_with_bounds)

# cv2.imshow("Show the words", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
