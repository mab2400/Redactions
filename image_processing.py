def pdf_to_jpg():
    """ Converts a multiple-page PDF into multiple single JPEG files """
    import os
    from pdf2image import convert_from_path

    pdf_dir = "/Users/miabramel/Downloads/pdbs"
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
    import pyautogui
    import cv2
    import imutils
    import numpy as np
    import math
    import time

    img = cv2.imread(pdf_file)
    img_original = cv2.imread(pdf_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.erode(gray, kernel, iterations=1)
    blur = cv2.GaussianBlur(img_erosion,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours and detect shape
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Identifying the Shape
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
            shape = 0
            peri = cv2.arcLength(c, True)
            # cv2.drawContours(thresh, c, -1, (0,255,0), 3)

            # smallest size of a redaction
            if peri > 700:
                # compute the bounding box of the contour
                approx = cv2.approxPolyDP(c, 0.04*peri, True)
                (x, y, w, h) = cv2.boundingRect(approx)

                # if the redaction is oddly shaped
                if w >= 7 and h >= 7 and  x != 0 and y != 0:
                    print("Irregularly shaped redaction found.")
                    shape = x, x+w, y, y+h
                    potential.append((shape, c))
                    redactions.append(c)
                    cv2.putText(img, "REDACTION", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (36,255,12), 10)

                # if the redaction is a perfect rectangle
                elif len(approx) == 4:
                    print("Rectangular redaction found.")
                    shape = x, x+w, y, y+h
                    potential.append((shape, c))
                    redactions.append(c)
                    cv2.putText(img, "REDACTION", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (36,255,12), 10)

    print("Count: ", len(redactions))
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", thresh)

    for (shape, contour) in potential:
        roi = thresh[shape[2]:shape[3], shape[0]:shape[1]]
        non_zero = np.count_nonzero(roi)

        # Maybe we should change > 0.95. Usually it's 0.3 or less.
        # if (non_zero/roi.size) > 0.95:
        next_potential.append((shape, contour))

    # final_redactions contains (shape, contour) objects
    final_redactions = isOverlapping(next_potential)

    for item in final_redactions:
        # item is (shape, contour)
        # To extract the contour part (which can be of any size), I set contour to item[1]
        contour = item[1]
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print("PUTTING REDACTION")
            cv2.putText(img, "REDACTION", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (36,255,12), 10)

    if len(final_redactions) < 24:
        cv2.imshow("Image", img)
    else:
        cv2.imshow("Map", img_original)

    """
    slash = pdf_file.rindex("/")
    underscore = pdf_file.rindex("-")
    period = pdf_file.rindex(".")
    docid = pdf_file[slash+1:underscore]
    pagenum = pdf_file[underscore+5:period]
    """

    if len(final_redactions) != 0:
        print("insdie final_redactions")
        for r in final_redactions:
            """
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
            """

    cv2.waitKey()

    '''
    # Pressing a key to exit will automatically take a screenshot
    slash = pdf_file.rindex("/")
    period = pdf_file.rindex(".")
    screenshot_name = pdf_file[slash + 1:period] + "-screenshot" + pdf_file[period:]
    pyautogui.screenshot(screenshot_name)
    print("Screenshot saved as ", screenshot_name)
    '''

    return ret

def isOverlapping(next_potential):

    # next_potential is of the form [(shape, contour), (shape, contour), ... ]
    # We want to return a similarly formatted list of only the non-overlapping redactions.

    redactions = [shape for (shape, contour) in next_potential]

    print(redactions)

    final_redactions_shape_contour = []
    if len(redactions) == 0:
        return redactions
    else:
        for i in range(0, len(redactions)-1):
            for j in range(i+1, len(redactions)):
                if (redactions[i][0] + 2 >= redactions[j][0] and redactions[j][0] >= redactions[i][0] - 2
                    and redactions[i][1] + 2 >= redactions[j][1] and redactions[j][1] >= redactions[i][1] - 2
                    and redactions[i][2] + 2 >= redactions[j][2] and redactions[j][2] >= redactions[i][2] - 2
                    and redactions[i][2] + 2 >= redactions[j][2] and redactions[j][2] >= redactions[i][2] - 2):

                    # Adding the (shape, contour) of the redaction
                    final_redactions_shape_contour.append(next_potential[i])
                else:
                    print("overlapping redaction found")

    # ret contains (shape, contour) items
    ret = [red for red in redactions if red not in final_redactions_shape_contour]

    if len(final_redactions_shape_contour) == 0:
        return redactions
    else:
        return ret

# pdf_to_jpg()
image_processing('/Users/miabramel/Downloads/pdbs/DOC_0005958912-page4.jpg')
