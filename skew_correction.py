import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
import datefinder
from dateutil.parser import parse

def apply_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.float32) / 15
    filtered = cv2.filter2D(gray, -1, kernel)
    # plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
    # plt.title('Filtered Image')
    # plt.show()
    return filtered

def apply_threshold(filtered):
    ret, thresh = cv2.threshold(filtered, 250, 255, cv2.THRESH_OTSU)
    # plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
    # plt.title('After applying OTSU threshold')
    # plt.show()
    return thresh

def detect_contour(img, image_shape):
    canvas = np.zeros(image_shape, np.uint8)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(canvas, cnt, -1, (0, 255, 255), 3)
    # plt.title('Largest Contour')
    # plt.imshow(canvas)
    # plt.show()

    return canvas, cnt

def detect_corners_from_contour(canvas, cnt):
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())
    # print('\nThe corner points are ...\n')
    # for index, c in enumerate(approx_corners):
    #     character = chr(65 + index)
    #     print(character, ':', c)
    #     cv2.putText(canvas, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Rearranging the order of the corner points
    approx_corners = [approx_corners[i] for i in [1, 2, 3, 0]]

    # plt.imshow(canvas)
    # plt.title('Corner Points: Douglas-Peucker')
    # plt.show()
    return approx_corners

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def warp_image(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    filtered_image = apply_filter(image)
    threshold_image = apply_threshold(filtered_image)
    cnv, largest_contour = detect_contour(threshold_image, image.shape)
    corners = detect_corners_from_contour(cnv, largest_contour)
    pts = np.array(corners, dtype = "float32")
    warped = four_point_transform(image, pts)
    return warped
    
def contrast_brightness(image):
    contrast = np.zeros(image.shape,image.dtype)
    alpha=1.1
    beta=-20
    #contrast=[contrast[y,x,c]=np.clip(alpha*image[y,x,c] + beta,0,255) for y in range(image.shape[0]) for x in range(image.shape[1]) for c in range(image.shape[2])]
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                contrast[y,x,c]=np.clip(alpha*image[y,x,c] + beta,0,255)
    
    return contrast

def threshold_remove_this_later(image):
    yen_threshold = threshold_yen(image)
    bright = rescale_intensity(image, (0, yen_threshold), (0, 255))
    return bright

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def fuzzy_date(text):
    for s in text.split():
        try:
            print(parse(s))
        except ValueError:
            pass
    # due_dates = datefinder.find_dates(text)
    # for match in due_dates:
    #     print(match)

if __name__ == '__main__':
    image = cv2.imread('receipt.jpg')
    final = warp_image(image)
    final = contrast_brightness(final)
    final = unsharp_mask(final)
    extracted_text1 = pytesseract.image_to_string(final)
    print(extracted_text1)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    extracted_text2 = pytesseract.image_to_string(image)
    print(extracted_text2)
    print("vvvvvvvvvvvvvvvvvvvvv")
    fuzzy_date(str(extracted_text1))
    print("vvvvvvvvvvvvvvvvvvvvv")
    fuzzy_date(str(extracted_text2))
    # cv2.imshow("Original", image)
    # cv2.imshow("Modified", final)
    # cv2.waitKey(0)
    