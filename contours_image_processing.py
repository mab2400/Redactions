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

    kernel = np.ones((10,10), np.uint8)
    img_erosion = cv2.erode(gray, kernel, iterations=1)
    blur = cv2.GaussianBlur(img_erosion,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    ret = []
    redactions = []
    potential = []
    next_potential = []
    total_area = 0

    for c in contours:

        shape_cX_cY_approx = get_shape_cX_cY_approx(c)

        if len(shape_cX_cY_approx) != 0:
            shape = shape_cX_cY_approx[0]
            approx = shape_cX_cY_approx[2]
            if shape != 0:
                x = shape[0]
                w = shape[1] - x
                y = shape[2]
                h = shape[3] - y

                # if the redaction is oddly shaped
                if w >= 7 and h >= 7 and  x != 0 and y != 0:
                    print("Irregularly shaped redaction found.")
                    potential.append(shape)
                    redactions.append(c)

                # if the redaction is a perfect rectangle
                elif len(approx) == 4:
                    print("Rectangular redaction found.")
                    potential.append(shape)
                    redactions.append(c)

    print("Count: ", len(redactions))
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", thresh)

    for shape in potential:
        roi = thresh[shape[2]:shape[3], shape[0]:shape[1]]
        non_zero = np.count_nonzero(roi)

        # Maybe we should change > 0.95. Usually it's 0.3 or less.
        # if (non_zero/roi.size) > 0.95:
        next_potential.append(shape)

    # ----------------------------------------------------------------------------------
    # At this point, redactions contains contours and next_potential contains shapes.

    # Step 1) Check if any contours are overlapping by calling contourIntersect on redactions.

    # Making a copy of redactions, from which we'll remove the intersecting contours.
    redactions_non_intersect = list(redactions)

    for i in range(0, len(redactions)-1):
        for j in range(i+1, len(redactions)):
            contour1 = redactions[i]
            contour2 = redactions[j]
            # if contourIntersect(img, contour1, contour2):
            if contour_intersect(contour1, contour2, True):
                print("Overlapping found")
                # Remove only one of the intersecting contours, just like removing a duplicate.
                redactions_non_intersect.remove(contour1)

    # Step 2) Call putText on the non-intersecting contours and convert the non-intersecting contours
    # to shapes, populating final_redactions.

    final_redactions = []

    for contour in redactions_non_intersect:
        # Note: shape_cX_cY_approx looks like [(x, x+w, y, y+h), (cX, cY), approx]
        shape_cX_cY_approx = get_shape_cX_cY_approx(contour)

        if len(shape_cX_cY_approx) > 0:

            shape = shape_cX_cY_approx[0]
            cX = shape_cX_cY_approx[1][0]
            cY = shape_cX_cY_approx[1][1]

            # Add to final_redactions
            final_redactions.append(shape)
            # Print "REDACTION" on the image
            cv2.putText(img, "REDACTION", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (36,255,12), 10)

    if len(final_redactions) < 24:
        # Testing out drawing ALL the contours, including overlapping ones.
        cv2.drawContours(img, redactions, 2, (255, 0, 0), 3)
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
        print("inside final_redactions")
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

def get_shape_cX_cY_approx(contour):
    import cv2

    # Returns: [shape, (cX, cY), approx] when given a contour.
    # If the perimeter is < 700, shape_cX_cY_approx will be []

    shape_cX_cY_approx = []
    M = cv2.moments(contour)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        peri = cv2.arcLength(contour, True)

        if peri > 700:

            # compute the bounding box of the contour
            approx = cv2.approxPolyDP(contour, 0.04*peri, True)
            (x, y, w, h) = cv2.boundingRect(approx)
            shape_cX_cY_approx = [(x, x+w, y, y+h), (cX, cY), approx]

    return shape_cX_cY_approx

def contour_intersect(contour1, contour2, edges_only = True):
    import cv2

    intersecting_pts = []

    ## Loop through all points in the contour
    for pt in contour2:
        x,y = pt[0]

        # Find points that intersect contour1
        # edges_only flag checks if the intersection to detect is only at the edges of the contour

        if edges_only and (cv2.pointPolygonTest(contour1,(x,y),True) == 0):
            intersecting_pts.append(pt[0])
        elif not(edges_only) and (cv2.pointPolygonTest(contour1,(x,y),True) >= 0):
            intersecting_pts.append(pt[0])

    if len(intersecting_pts) > 0:
        return True
    else:
        return False

def contourIntersect(original_image, contour1, contour2):
    import numpy as np
    import cv2

    # Two separate contours trying to check intersection on
    contours = [contour1, contour2]

    # Create image filled with zeros the same size of original image
    blank = np.zeros(original_image.shape[0:2])

    # Copy each contour into its own image and fill it with '1'
    image1 = cv2.drawContours(blank.copy(), contours, 0, 1)
    image2 = cv2.drawContours(blank.copy(), contours, 1, 1)

    # Use the logical AND operation on the two images
    # Since the two images had bitwise and applied to it,
    # there should be a '1' or 'True' where there was intersection
    # and a '0' or 'False' where it didnt intersect
    intersection = np.logical_and(image1, image2)

    # Check if there was a '1' in the intersection
    return intersection.any()

def mia_isOverlapping(next_potential):
    pass
    # next_potential consists of shapes

    # shape = ...

    # for i in range(0, len(tuple_list)-1):
         # for j in range(i+1, len(tuple_list)):
             # Shapes are at index[i or j][0]
             # Check if tuple_list[1][0]

# pdf_to_jpg()
image_processing('/Users/miabramel/Downloads/pdbs/DOC_0005958918-page2.jpg')
