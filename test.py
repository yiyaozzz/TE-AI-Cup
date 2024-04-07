import cv2
import numpy as np
import os
img = cv2.imread('img10.png')


def thresholding(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.GaussianBlur(thresh, (7, 7), 0)
    ret, thresh = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)
    return thresh


thresh_img = thresholding(img)
cv2.imshow("thresh_img", thresh_img)

linesArray = []
kernelRows = np.ones((5, 35), np.uint8)

dilated = cv2.dilate(thresh_img, kernelRows, iterations=2)
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


kernelWords = np.ones((7, 10), np.uint8)
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


# import cv2
# import numpy as np

# img = cv2.imread('img11.png')


# def thresholding(image):
#     img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     ret, thresh = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY_INV)
#     thresh = cv2.GaussianBlur(thresh, (7, 7), 0)  # change values
#     ret, thresh = cv2.threshold(
#         thresh, 100, 255, cv2.THRESH_BINARY)  # Change values
#     return thresh


# thresh_img = thresholding(img)
# cv2.imshow("thresh_img", thresh_img)

# # line sementation

# linesArray = []
# kernelRows = np.ones((5, 35), np.uint8)


# dilated = cv2.dilate(thresh_img, kernelRows, iterations=2)
# cv2.imshow("dilated", dilated)

# # find contours
# (contoursRows, heirarchy) = cv2.findContours(
#     dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cv2.drawContours(img, contoursRows, -1 , (0,255,0), 2) # draw the contour around each row on the original image

# # loop inside the contours and draw rectangle
# for row in contoursRows:
#     area = cv2.contourArea(row)
#     if area > 500:
#         x, y, w, h = cv2.boundingRect(row)
#         # cv2.rectangle(img, (x,y), (x+w, y+h), (40,100,250), 2)
#         linesArray.append([x, y, w, h])

# print(len(linesArray))

# sortedLinesArray = sorted(linesArray, key=lambda line: line[1])

# words = []
# lineNumber = 0
# all = []


# kernelWords = np.ones((7, 10), np.uint8)
# dilateWordsImg = cv2.dilate(thresh_img, kernelWords, iterations=1)
# cv2.imshow("dilate Words Img", dilateWordsImg)

# for line in sortedLinesArray:

#     x, y, w, h = line

#     roi_line = dilateWordsImg[y: y+h, x:x+w]
#     (contoursWords, heirarchy) = cv2.findContours(
#         roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for word in contoursWords:
#         x1, y1, w1, h1 = cv2.boundingRect(word)
#         # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2 )
#         cv2.rectangle(img, (x+x1, y+y1), (x+x1+w1, y+y1+h1),
#                       (255, 255, 0), 2)  # Change values
#         words.append([x+x1, y+y1, x+x1+w1, y+y1+h1])

#     sortedWords = sorted(words, key=lambda line: line[0])  # (x,y,w,h)

#     for word in sortedWords:
#         a = (lineNumber, word)
#         all.append(a)

#     lineNumber = lineNumber + 1

# chooseWord = all[3][1]
# print(chooseWord)

# roiWord = img[chooseWord[1]:chooseWord[3], chooseWord[0]:chooseWord[2]]
# cv2.imshow("Show a word", roiWord)


# cv2.imshow("Show the words", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
