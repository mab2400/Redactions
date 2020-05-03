
def pdf_to_jpg():

	#conda install -c conda-forge poppler
	#pip install pdf2image

	# from pdf2image import convert_from_path

	# from pdf2image.exceptions import (
 #    PDFInfoNotInstalledError,
 #    PDFPageCountError,
 #    PDFSyntaxError
	# )

	# ret = []

	# images = convert_from_path('/Users/carriehaykellar/Downloads/DOC_0005976579.pdf', output_folder='/Users/carriehaykellar/Desktop/test1')

	# ret.append(images)
	# print(ret)

	import os
	from pdf2image import convert_from_path

	pdf_dir = "/Users/carriehaykellar/Downloads/PDFs"
	os.chdir(pdf_dir)
	for pdf_file in os.listdir(pdf_dir):
	    if pdf_file.endswith(".pdf"):
	        pages = convert_from_path(pdf_file, 300)
	        pdf_file = pdf_file[:-4]
	        for page in pages:
	           page.save("%s-page%d.jpg" % (pdf_file,pages.index(page)), "JPEG")

	return images


def image_processing():

	import cv2
	import imutils
	import numpy as np

	img = cv2.imread('/Users/carriehaykellar/Desktop/test1/cf5aadb3-94e3-4c05-a7b1-663bc4e2edc9-16.ppm')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# ret, thresh = cv2.threshold(gray, 127,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
	

	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# contours = imutils.grab_contours(contours)	

	total_area = 0
	for c in contours:
		M = cv2.moments(c)
		if M["m00"] != 0:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			shape= detect(c)
			if shape == 'redaction':
				area = cv2.contourArea(c)
				total_area += area
				print("Area:" , area)

				#cv2.rectangle(edges, (x,y), (x+w, y+h), (225, 225, 225), 2)
				cv2.drawContours(thresh, [c], 0, (0,255,0), 1)
				cv2.putText(thresh, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)

	#area = cv2.contourArea(contours)
	frame = thresh.size
	print( 'Original Dimensions : ', frame)
	print("total area:", total_area)
	perc = total_area/frame
	print("percentage of redactions:", perc * 100)

	cv2.imshow('binary_image', thresh)
	cv2.waitKey()
	cv2.destroyAllWindows()


def detect(c):
	import cv2
	
	shape = "unknown"
	peri = cv2.arcLength(c, True)
	rectanges = []
	if peri > 300:
		approx = cv2.approxPolyDP(c, 0.04*peri, True)


	# if the shape has 4 vertices, it is either a square or
	# a rectangle
		if len(approx) == 4 and len(c) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			if x != 0 and y != 0:
				rectanges.append([x, y, w, h])
				print(x, y, w, h)
				shape = "redaction"
				print("Perimeter: ", peri)
				print(shape)


		return shape


pdf_to_jpg()
# image_processing()