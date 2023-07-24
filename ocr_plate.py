import cv2
import pytesseract
import numpy as np
import imutils
from skimage.segmentation import clear_border


def debug_imshow(title, image):
	cv2.imshow(title, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def preProcessing(img): 
    img = cv2.equalizeHist(img)
    return img
def isNearest(h,w):
	if h > 100:
		if w > 300:
			return True
		else :
			return False
	return None

def locate_license_plate_candidates(gray, keep=20):
		# perform a blackhat morphological operation that will allow
		# us to reveal dark regions (i.e., text) on light backgrounds
		# (i.e., the license plate itself)
		
		
		rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13,7))
		blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

		squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
		light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
		light = cv2.threshold(light, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

		

		gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
			dx=1, dy=0, ksize=-1)
		gradX = np.absolute(gradX)
		(minVal, maxVal) = (np.min(gradX), np.max(gradX))
		gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
		gradX = gradX.astype("uint8")


		gradX = cv2.GaussianBlur(gradX, (7, 7), 0)
		gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
		thresh = cv2.threshold(gradX, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

		

		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=4)

		thresh = cv2.bitwise_and(thresh, thresh, mask=light)
		thresh = cv2.dilate(thresh, None, iterations=6)
		thresh = cv2.erode(thresh, None, iterations=1)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
		
		# return the list of contours
		return cnts


def locate_license_plate(gray, candidates,clearBorder= True):
	# initialize the license plate contour and ROI
	lpCnt = None
	roi = None

	equal = preProcessing(gray)
	kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
	morph_image=cv2.morphologyEx(equal,cv2.MORPH_OPEN,kernel,iterations=10)
	sub_morp_image=cv2.subtract(equal,morph_image)
	for c in candidates:
		box = cv2.boxPoints(cv2.minAreaRect(c))
		box = box.astype("int")
		peri = cv2.arcLength(box,True)
		approx = cv2.approxPolyDP(box,0.05 * peri,True)
		if len(approx) == 4:
			lpCnt = c
			(x, y, w, h) = cv2.boundingRect(c)
			# if x < 0:
			# 	x *= -1
			# if y < 0:
			# 	y *= -1
			ar = float(w/h)
			isNear = isNearest(h,w)
			if isNear:
				licensePlate = gray[y:y + h, x:x + w]
				licensePlate = cv2.resize(licensePlate,(256,256))
				thresh = cv2.threshold(licensePlate, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
				thresh = cv2.erode(thresh,None,iterations=1)
				roi = clear_border(thresh)

				break
			elif isNear == None:
				if ar > 1.45:
					licensePlate = gray[y:y + h, x:x + w]
					licensePlate = cv2.resize(licensePlate,(256,256))
					thresh = cv2.threshold(licensePlate, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
					thresh = cv2.erode(thresh,None,iterations=1)
					roi = clear_border(thresh)
					break
				


	

	return (roi, lpCnt)

def ocr(gray):
	gray = cv2.bilateralFilter(gray,9,75,75)
	cnts = locate_license_plate_candidates(gray)
	(lp, lpCnt) = locate_license_plate(gray,cnts)
	return (lp, lpCnt)