import os
import csv
import cv2
import time
import math
import imutils
import pyautogui
import statistics
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pdf2image import convert_from_path

def pdf_to_jpg(pdf_dir):
    """ Converts a multiple-page PDF into multiple single JPEG files """
    # pdf_dir looks like "/Users/miabramel/Downloads/pdbs"
    os.chdir(pdf_dir)
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pages = convert_from_path(pdf_file, 300)
            pdf_file = pdf_file[:-4]
            for page in pages:
               page.save("%s-page%d.jpg" % (pdf_file,pages.index(page)), "JPEG")

def unpickle():
    unpickled_df = pd.read_pickle("/Users/carriehaykellar/Downloads/pdfs-2/dummy.pkl")
    print(unpickled_df)
    engine = create_engine('mysql://dbuser:dbuserdbuser@127.0.0.1:3306/redactions')
    unpickled_df.to_sql('redactions_test', con=engine)

def process_files():
    """ Calls image_processing on each JPEG file """
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

def putRedactions(redaction_shapes, img):
    """ Writes REDACTION on top of the image. """
    for shape in redaction_shapes:
        bottom_left_corner = (int(shape[0]), int(shape[3] - 15))
        top_left_corner = (int(shape[0]), int(shape[2]))
        bottom_right_corner = (int(shape[1]), int(shape[3]))
        cv2.putText(img, "REDACTION", bottom_left_corner, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (36,255,12), 10)

def drawTextRectangles(text_shapes, img):
    """ Draws the bounding rectangles of the detected text on the page. """
    for shape in text_shapes:
        top_left_corner = (int(shape[0]), int(shape[2]))
        bottom_right_corner = (int(shape[1]), int(shape[3]))
        cv2.rectangle(img, top_left_corner, bottom_right_corner, (255,0,0), 5)

def getPercentRedacted(redaction_shapes, text_shapes):
    """ Calculates the percent of text redacted. """
    total_redaction_rectangle_area = 0
    text_area = 0
    for shape in redaction_shapes:
        width = abs(shape[1] - shape[0])
        height = abs(shape[3] - shape[2])
        area = width * height
        total_redaction_rectangle_area += area
    for shape in text_shapes:
        width = abs(shape[1] - shape[0])
        height = abs(shape[3] - shape[2])
        area = width * height
        text_area += area
    # I calculated that text makes up 26% of the redaction rectangle area (total_redaction_rectangle_area).
    redacted_text_area = .26 * total_redaction_rectangle_area
    estimated_text_area = text_area + redacted_text_area
    return (redacted_text_area / estimated_text_area)

def take_screenshot(pdf_file):
    # Pressing a key to exit will automatically take a screenshot
    slash = pdf_file.rindex("/")
    period = pdf_file.rindex(".")
    screenshot_name = pdf_file[slash + 1:period] + "-screenshot" + pdf_file[period:]
    pyautogui.screenshot(screenshot_name)
    print("Screenshot saved as", screenshot_name)

def get_midpoint(shape):
    # Finds the midpoint of a shape when given (1169, 1648, 2405, 2469) for example.
    # The midpoints of two shapes will be compared to test whether they are overlapping.
    x_1 = shape[0]
    x_2 = shape[1]
    y_1 = shape[2]
    y_2 = shape[3]

    midpoint = ((x_1 + x_2)/2, (y_1 + y_2)/2)
    return midpoint

def get_euclidean_dist(shape1, shape2):
    # Calculates the distance between two midpoints of two shapes.
    midpoint1 = get_midpoint(shape1)
    midpoint2 = get_midpoint(shape2)

    midpoint1_x = midpoint1[0]
    midpoint1_y = midpoint1[1]
    midpoint2_x = midpoint2[0]
    midpoint2_y = midpoint2[1]

    dist = math.sqrt(math.pow((midpoint1_x - midpoint2_x),2) + math.pow((midpoint1_y - midpoint2_y),2))
    return dist

def get_non_overlapping_shapes(next_potential):
    # next_potential is a list of shapes: [shape1, shape2, shape3]

    # I make a copy of next_potential, removing duplicates (aka removing one of the overlapping
    # redactions) so we are left with a list of non-overlapping shapes

    final_redactions = list(next_potential)
    distances = []

    for i in range(0, len(next_potential)-1):
         for j in range(i+1, len(next_potential)):
             # Compare the shapes next_potential[i] and next_potential[j].
             shape1 = next_potential[i]
             shape2 = next_potential[j]
             midpoint_dist = get_euclidean_dist(shape1, shape2)
             # TODO: Change this from print() to simply if the distance is below a certain value
             # print("dist btwn midpoints = ", midpoint_dist)
             distances.append(midpoint_dist)

             # TODO: FIGURE OUT WHAT NUMBER THIS SHOULD ACTUALLY BE
             if midpoint_dist < 250:
                 # The shapes must be overlapping.
                 # We want only one of the two overlapping shapes to remain in the list.

                 # If BOTH overlapping shapes are in final_redactions, let's remove one of them.
                 if (next_potential[i] in final_redactions) and (next_potential[j] in final_redactions):
                     final_redactions.remove(next_potential[i])
                 # Otherwise, don't remove anything because that means there is already only ONE in the list.

    # Calculating statistics on the distances to get a sense of what the threshold should be:
    # print("smallest distance = ", min(distances))

    return final_redactions

def get_redaction_shapes_text_shapes(contours):
    potential = []
    text_potential = []

    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            shape = 0
            peri = cv2.arcLength(c, True)
            # cv2.drawContours(thresh, c, -1, (0,255,0), 3)

            if cY > 900 and cY < 3150:

                # Detecting the redaction
                if peri > 550 and peri < 9000:
                    # compute the bounding box of the contour
                    approx = cv2.approxPolyDP(c, 0.04*peri, True)
                    (x, y, w, h) = cv2.boundingRect(approx)

                    # if the redaction is oddly shaped
                    if w >= 7 and h >= 7 and  x != 0 and y != 0:
                        shape = x, x+w, y, y+h
                        potential.append(shape)
                        # redactions.append(c)

                    # if the redaction is a perfect rectangle
                    elif len(approx) == 4:
                        shape = x, x+w, y, y+h
                        potential.append(shape)
                        # redactions.append(c)

                # Detecting the text
                if peri > 25 and peri < 150:
                    approx = cv2.approxPolyDP(c, 0.04*peri, True)
                    (x, y, w, h) = cv2.boundingRect(approx)
                    shape = x, x+w, y, y+h
                    text_potential.append(shape)
                    # redactions.append(c)

    return (potential, text_potential)

def analyze_results(output_file):

    total_redaction_count = 0
    total_percent_redacted = 0
    document_count = 0

    with open(output_file, mode='r') as output:
        reader = csv.reader(output)
        for row in reader:
            document_count += 1
            total_redaction_count += int(row[0])
            total_percent_redacted += float(row[1])
    output.close()

    print("PDB Count: ", document_count)
    print("Average Redaction Count: ", int(total_redaction_count / document_count))
    print("Average Percent of Text Redacted: ", total_percent_redacted / document_count)
