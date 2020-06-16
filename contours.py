"""
import cv2
def detect_shape(c):
    # Compute perimeter of contour and perform contour approximation
    shape = ""
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # Rectangle
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # A square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    return shape

# Load image, grayscale, Otsu's threshold
image = cv2.imread('1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours and detect shape
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    # Identify shape
    shape = detect_shape(c)

    # Find centroid and label shape name
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.putText(image, shape, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)

cv2.imshow('thresh', thresh)
cv2.imshow('image', image)
cv2.waitKey()
"""

def pdf_to_jpg():
    """ Converts a multiple-page PDF into multiple single JPEG files """
    import os
    from pdf2image import convert_from_path

    pdf_dir = "/Users/carriehaykellar/Downloads/pdfs-2"
    os.chdir(pdf_dir)
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pages = convert_from_path(pdf_file, 300)
            pdf_file = pdf_file[:-4]
            for page in pages:
               page.save("%s-page%d.jpg" % (pdf_file,pages.index(page)), "JPEG")

def unpickle():
    import pandas as pd
    from sqlalchemy import create_engine
    unpickled_df = pd.read_pickle("/Users/carriehaykellar/Downloads/pdfs-2/dummy.pkl")
    print(unpickled_df)
    engine = create_engine('mysql://dbuser:dbuserdbuser@127.0.0.1:3306/redactions')
    unpickled_df.to_sql('redactions_test', con=engine)

def process_files():
    """ Calls image_processing on each JPEG file """
    import os
    import pandas as pd
    import numpy as np
    from sqlalchemy import create_engine
    import time

    t1 = time.time()
    df = pd.DataFrame(columns=['docid', 'pagenum', 'area', 'perc', 'aspect_ratio', 'upper_left_x', 'upper_left_y', 'bottom_left_x', 'bottom_left_y'])
    pdf_dir = "/Users/carriehaykellar/Downloads/pdfs-2"
    os.chdir(pdf_dir)
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".jpg"):
            filePath = pdf_dir +"/" + pdf_file
            print(filePath)
            redactions = image_processing(filePath)
            rdf = pd.DataFrame(data= redactions, columns=['docid', 'pagenum', 'area', 'perc', 'aspect_ratio', 'upper_left_x', 'upper_left_y', 'bottom_left_x', 'bottom_left_y'])
            df = df.append(rdf, ignore_index = True)
            df.to_pickle("./dummy.pkl")

    t2 = time.time()
    print("Run Time: ", round(t2-t1))
    engine = create_engine('mysql://dbuser:dbuserdbuser@127.0.0.1:3306/redactions')
    df.to_sql('raw_redactions', con=engine)

def linedetection(pdf_file):
    import time
    import cv2
    import imutils
    import numpy as np
    import math

    src = cv2.imread(pdf_file)
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.erode(src, kernel, iterations=1)
    blur = cv2.GaussianBlur(img_erosion,(5,5),0)
    cv2.imshow("Erosion", blur)

    dst = cv2.Canny(blur, 50, 200, None, 3)
    cv2.imshow("cannt", dst)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 20, 0)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            print(l)
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    # cv2.imshow("Source", src)
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    cv2.waitKey()

def image_processing(pdf_file):
    """ Finds contours """
    import cv2
    import imutils
    import numpy as np
    import math

    img = cv2.imread(pdf_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.erode(gray, kernel, iterations=1)
    blur = cv2.GaussianBlur(img_erosion,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    ret = []
    redactions = []
    potential = []
    next_potential = []
    total_area = 0
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            redaction = 0
            shape = 0
            peri = cv2.arcLength(c, True)

            cv2.drawContours(thresh, c, 2, (0,255,0), 3)

            # smallest size of a redaction
            if peri > 300:
                # compute the bounding box of the contour
                approx = cv2.approxPolyDP(c, 0.04*peri, True)
                (x, y, w, h) = cv2.boundingRect(approx)

                # if the redaction is oddly shaped
                if w >= 7 and h >= 7 and  x != 0 and y != 0:
                    redaction = 1
                    shape = x, x+w, y, y+h
                    potential.append(shape)

                # if the redaction is a perfect rectangle
                elif len(approx) == 4 and len(c) == 4 and x != 0 and y != 0:
                    redaction = 1
                    shape = x, x+w, y, y+h
                    potential.append(shape)

                else:
                    redaction = 0

    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", thresh)
    cv2.waitKey()

    for shape in potential:
        roi = thresh[shape[2]:shape[3], shape[0]:shape[1]]
        non_zero = np.count_nonzero(roi)

        if (non_zero/roi.size) > 0.95:
            next_potential.append(shape)

    final_redactions = isOverlapping(next_potential)

    slash = pdf_file.rindex("/")
    underscore = pdf_file.rindex("-")
    period = pdf_file.rindex(".")
    docid = pdf_file[slash+1:underscore]
    pagenum = pdf_file[underscore+5:period]

    if len(final_redactions) != 0:
        for r in final_redactions:
            area = (r[1]-r[0]) * (r[3]-r[2])
            start = (r[0],r[2])
            end = (r[1], r[3])
            frame = thresh.size
            margin = (2550-1950)*3301
            r_perc = round(area/(frame-margin) * 100, 2)
            aspect_ratio = round((r[1]-r[0])/(r[3]-r[2]), 2)
            upper_left_x = r[0]
            upper_left_y = r[2]
            bottom_left_x = r[1]
            bottom_left_y = r[3]
            r_info = [docid, pagenum, area, r_perc, aspect_ratio, upper_left_x, upper_left_y, bottom_left_x, bottom_left_y]
            ret.append(r_info)
    else:
        r_info = [docid, pagenum, 0, 0, 0, 0, 0, 0, 0]
        ret.append(r_info)

    return ret

def isOverlapping(redactions):

    final_redactions = []
    if len(redactions) == 0:
        return redactions
    else:
        for i in range(0, len(redactions)-1):
            for j in range(i+1, len(redactions)):
                if (redactions[i][0] + 2 >= redactions[j][0] and redactions[j][0] >= redactions[i][0] - 2
                    and redactions[i][1] + 2 >= redactions[j][1] and redactions[j][1] >= redactions[i][1] - 2
                    and redactions[i][2] + 2 >= redactions[j][2] and redactions[j][2] >= redactions[i][2] - 2
                    and redactions[i][2] + 2 >= redactions[j][2] and redactions[j][2] >= redactions[i][2] - 2):
                    final_redactions.append(redactions[i])

    ret = [red for red in redactions if red not in final_redactions]

    if len(final_redactions) == 0:
        return redactions
    else:
        return ret

linedetection('/Users/miabramel/Downloads/test.jpg')
