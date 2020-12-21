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
from collections import Counter
from sqlalchemy import create_engine
from pdf2image import convert_from_path

def convert_all_pdfs_to_jpgs(pdf_dir):
    """ Converts ALL multi-page PDFs in the directory into multiple single JPEG files """
    # pdf_dir looks like "/Users/miabramel/Downloads/pdbs"
    os.chdir(pdf_dir)
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pages = convert_from_path(pdf_file, 300)
            pdf_file = pdf_file[:-4]
            for page in pages:
               page.save("%s-page%d.jpg" % (pdf_file,pages.index(page)), "JPEG")

def pdf_to_jpg(pdf_dir, pdf_file):
    """ Converts a multiple-page PDF into multiple single JPEG files """
    # pdf_dir looks like "/Users/miabramel/Downloads/pdbs" as an example
    os.chdir(pdf_dir)
    full_name = pdf_dir + "/" + pdf_file
    pages = convert_from_path(full_name, 300)
    full_name = full_name[:-4]
    for page in pages:
        page.save("%s-page%d.jpg" % (full_name, pages.index(page)), "JPEG")

def pdf_to_png(pdf_dir, pdf_file):
    """ Converts a multiple-page PDF into multiple single PNG files """
    # pdf_dir looks like "/Users/miabramel/Downloads/pdbs" as an example
    os.chdir(pdf_dir)
    full_name = pdf_dir + "/" + pdf_file
    pages = convert_from_path(full_name, 300)
    full_name = full_name[:-4]
    for page in pages:
        page.save("%s-page%d.png" % (full_name, pages.index(page)), format="png")

def putRedactions(redaction_shapes, img):
    """ Writes REDACTED on top of the image. """
    for shape in redaction_shapes:
        bottom_left_corner = (int(shape[0]), int(shape[3] - 15))
        top_left_corner = (int(shape[0]), int(shape[2]))
        bottom_right_corner = (int(shape[1]), int(shape[3]))
        cv2.putText(img, "REDACTED", bottom_left_corner, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (36,255,12), 10)

def put_type1_redactions(redaction_shapes, img):
    for shape in redaction_shapes:
        bottom_left_corner = (int(shape[0]), int(shape[3] - 15))
        top_left_corner = (int(shape[0]), int(shape[2]))
        bottom_right_corner = (int(shape[1]), int(shape[3]))
        cv2.putText(img, "TYPE 1", bottom_left_corner, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255), 10)

def put_type2_redactions(redaction_shapes, img):
    for shape in redaction_shapes:
        bottom_left_corner = (int(shape[0]), int(shape[3] - 15))
        top_left_corner = (int(shape[0]), int(shape[2]))
        bottom_right_corner = (int(shape[1]), int(shape[3]))
        cv2.putText(img, "TYPE 2", bottom_left_corner, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255), 10)

def put_type3_redactions(redaction_shapes, img):
    for shape in redaction_shapes:
        bottom_left_corner = (int(shape[0]), int(shape[3] - 15))
        top_left_corner = (int(shape[0]), int(shape[2]))
        bottom_right_corner = (int(shape[1]), int(shape[3]))
        cv2.putText(img, "TYPE 3", bottom_left_corner, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255), 10)

def drawRedactionRectangles(redaction_shapes, img):
    """ Draws the bounding rectangles of the detected redactions on the page. """
    for shape in redaction_shapes:
        top_left_corner = (int(shape[0]), int(shape[2]))
        bottom_right_corner = (int(shape[1]), int(shape[3]))
        cv2.rectangle(img, top_left_corner, bottom_right_corner, (255,0,0), 5)

def drawTextRectangles(text_shapes, img):
    """ Draws the bounding rectangles of the detected text on the page. """
    for shape in text_shapes:
        top_left_corner = (int(shape[0]), int(shape[2]))
        bottom_right_corner = (int(shape[1]), int(shape[3]))
        cv2.rectangle(img, top_left_corner, bottom_right_corner, (255,0,0), 5)

def get_stats(redaction_shapes, text_shapes):
    """ Returns:
        1) the estimated area of text redacted on the page
        2) the total text area on the page
        3) the estimated number of words redacted """
    total_redaction_rectangle_area = 0
    text_area = 0
    # Summing up total redaction area
    for shape in redaction_shapes:
        width = abs(shape[1] - shape[0])
        height = abs(shape[3] - shape[2])
        area = width * height
        total_redaction_rectangle_area += area
    # Summing up total text area (not including redactions)
    for shape in text_shapes:
        width = abs(shape[1] - shape[0])
        height = abs(shape[3] - shape[2])
        area = width * height
        text_area += area
    # I calculated that text makes up 26% of the redaction rectangle area (total_redaction_rectangle_area).
    redacted_text_area = .26 * total_redaction_rectangle_area
    estimated_text_area = text_area + redacted_text_area # This includes maps
    # Now, I will divide redacted_text_area by the (estimated) area of a single word in order to estimate
    # the number of words redacted.
    estimated_word_area = 4665
    estimated_num_words = int(redacted_text_area / estimated_word_area)
    return [redacted_text_area, estimated_text_area, estimated_num_words]

def take_screenshot(pdf_file):
    # Pressing a key to exit will automatically take a screenshot
    slash = pdf_file.rindex("/")
    period = pdf_file.rindex(".")
    screenshot_name = pdf_file[slash + 1:period] + "-screenshot" + pdf_file[period:]
    pyautogui.screenshot(screenshot_name)
    print("Screenshot saved as", screenshot_name)

def get_intersection_over_union(potential):
    import cv2
    """ Returns a list of non-overlapping redaction shapes.
    Note that if the page contained a map, no redactions will be returned."""

    rejects = []
    is_map = False
    final_redactions = []

    # Iterates through potential redaction list
    for boxA in range(0, len(potential)-1):
        for boxB in range(boxA+1, len(potential)):
            iou = getIOU(potential[boxA], potential[boxB])

            # if redactions overlap, append to reject list
            if iou > 0.001:
                rejects.append(potential[boxA])

    if len(rejects) > 0 and Counter(rejects).most_common(1)[0][1] > 4:
        is_map = True

    final_redactions = [x for x in potential if x not in rejects]

    return final_redactions

def getIOU(boxA, boxB):
    """
    Determines the score of whether two redactions are overlapping.
    The following code is from
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[1], boxB[1])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1)
    boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1)

    # Compute the intersection over union by taking the intersection
    # area and divide it by the sum of prediction + ground-truth
    # areas - the intersection area
    denominator = float(boxAArea + boxBArea - interArea)
    if denominator == 0:
        iou = 0
    else:
        iou = interArea / denominator
    # return the intersection over union value
    return iou

def pdb_get_redaction_shapes_text_shapes(contours, img):
    potential = []
    text_potential = []

    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            shape = 0
            peri = cv2.arcLength(c, True)

            if cY > 15 and cY < 3150:

                # Detecting the redaction
                if peri > 550 and peri < 100000:
                    # compute the bounding box of the contour
                    approx = cv2.approxPolyDP(c, 0.04*peri, True)
                    (x, y, w, h) = cv2.boundingRect(approx)
                    i = np.array(img)
                    bounding = i[y:y+h+1, x:x+w+1]

                    non_zero = np.count_nonzero(bounding)
                    # Determine that contour is 95% white space
                    if (non_zero / bounding.size) > .95:
                        # Append to potential list if redaction meets criteria
                        # If the redaction is oddly shaped
                        if w >= 10 and h >= 10 and  x != 0 and y != 0:
                            shape = x, x+w, y, y+h
                            potential.append(shape)

                # Detecting the text
                if peri > 25 and peri < 150:
                    approx = cv2.approxPolyDP(c, 0.04*peri, True)
                    (x, y, w, h) = cv2.boundingRect(approx)
                    shape = x, x+w, y, y+h
                    text_potential.append(shape)

    (type1, type2, type3) = get_redaction_types(potential, img)

    return (potential, text_potential, type1, type2, type3)

def cib_get_redaction_shapes_text_shapes(contours, img):
    potential = []
    text_potential = []

    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            peri = cv2.arcLength(c, True)

            if peri > 500:
                # Detecting the redaction
                # Compute the bounding box of the contour
                approx = cv2.approxPolyDP(c, 0.04*peri, True)
                (x, y, w, h) = cv2.boundingRect(approx)
                if w*h > 6000000:
                    # Handling the case in which the script treats the entire outline of the page as a redaction
                    continue
                shape = x, x+w, y, y+h
                area = w*h
                i = np.array(img)
                bounding = i[y:y+h+1, x:x+w+1]

                if x > 1448 and x < 1942 and y > 25 and y < 200 and h > 50 and w > 200:
                    # Getting the one redaction at the top of the page with words within it.
                    potential.append(shape)
                else:
                    # Determine the percentage of white space
                    # RGB 0,0,0 is BLACK --> non_zero is WHITE
                    white = np.count_nonzero(bounding)
                    percent_white = white / bounding.size
                    if (percent_white > .80 and area < 100000) or (percent_white > .85 and area < 1000000) or (percent_white > .95 and area > 1000000):
                        if w < 3250 and w > 20 and h < 3250 and h > 20:
                            # Making sure it's within the margins of the image
                            potential.append(shape)

            # Detecting the text
            if peri > 25 and peri < 150:
                approx = cv2.approxPolyDP(c, 0.04*peri, True)
                (x, y, w, h) = cv2.boundingRect(approx)
                shape = x, x+w, y, y+h
                text_potential.append(shape)

    (type1, type2, type3) = get_redaction_types(potential, img)

    return (potential, text_potential, type1, type2, type3)

def get_redaction_types(potential, img):
    # potential is a list of shapes (x, x+w, y, y+h)
    type1 = []
    type2 = []
    type3 = []

    for (a, b, c, d) in potential:
        x = a
        w = b-a
        y = c
        h = d-c

        # TYPE 3: TOP/BOTTOM MARGIN
        if y < 200 or y > 2609:
            type3.append((a,b,c,d))
            continue

        # TYPE 2: LEFT/RIGHT MARGIN
        if x < 484 or x > 2101 and w*h > 1000:
            type2.append((a,b,c,d))
            continue

        # TYPE 1: END OF A PARAGRAPH ("Page X" after the redaction, and extra whitespace under it)
        # Create a box of the same size as the redaction, but shift it down by h/2 pixels
        x1 = x
        w1 = w
        h1 = h
        y1 = y+int(h/2)
        i = np.array(img)
        bounding = i[y1:y1+h1+1, x1:x1+w1+1]
        non_zero = np.count_nonzero(bounding)
        # Determine that the box is 95% white space
        if h < 90 and (non_zero / bounding.size) > .50:
            type1.append((a,b,c,d))

    return (type1, type2, type3)

def analyze_results(output_file):
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline, BSpline

    doc_count = 0
    total_redaction_count = 0
    total_percent_text_redacted = 0
    total_num_words_redacted = 0

    # I will assume the max number of redactions in a document is 1000.
    redaction_num_to_freq = {i:0 for i in range(1000)}

    # I know the percents will go from 0 to 100, I make the steps 0.5
    percent_range = [percent*(0.5) for percent in range(200)]
    percent_range.append(100)
    percent_to_freq = {j:0 for j in percent_range}

    name_to_percent = {}

    list_of_redaction_nums = []

    num_words_redacted_x = []
    num_redactions_y = []

    with open(output_file, mode='r') as output:
        reader = csv.reader(output)
        for row in reader:

            name = row[0][5:12]
            redaction_num = int(row[1])
            num_redactions_y.append(redaction_num)
            percent_text_redacted = float(row[2])
            num_words_redacted = int(row[3])
            num_words_redacted_x.append(num_words_redacted)
            list_of_redaction_nums.append(num_words_redacted)

            redaction_num_to_freq[redaction_num] += 1
            name_to_percent[name] = percent_text_redacted
            # rounds the percent to the nearest .5
            rounded_percent = round(percent_text_redacted * 200) / 2
            percent_to_freq[rounded_percent] += 1

            total_redaction_count += redaction_num
            total_percent_text_redacted += percent_text_redacted
            total_num_words_redacted += num_words_redacted

            doc_count += 1

    # Making a list of 1, 2, 3, ... doc_count
    num_words_keys = [i for i in range(0, doc_count+1)]
    list_of_redaction_nums.sort()
    # Making a dictionary to be used for a graph
    num_words_dict = {num1:num2 for (num1, num2) in zip(num_words_keys, list_of_redaction_nums)}

    output.close()

    dates = []

    with open("cib_meta.csv", mode='r') as output:
        reader = csv.reader(output)
        row_count = 0
        for row in reader:
            if row_count == 0:
                row_count+=1
                continue
            entire_date = row[12].split()
            if len(entire_date) > 0:
                date = int(entire_date[-1])
                dates.append(date)
            row_count+=1

    output.close()

    date_to_percents = {date:[] for date in dates}

    with open("cib_meta.csv", mode='r') as output:
        reader = csv.reader(output)
        row_count = 0
        for row in reader:
            if row_count == 0:
                row_count+=1
                continue
            name = row[4][1:]
            entire_date = row[12].split()
            if len(entire_date) > 0:
                date = int(entire_date[-1])
                percent = 0
                for key in name_to_percent.keys():
                    if key == name:
                        percent = name_to_percent[key]
                        break
                date_to_percents[date].append(percent)
            row_count+=1

    output.close()

    date_to_avg_percent = {}

    for date in date_to_percents.keys():
        avg_percent = sum(date_to_percents[date]) / len(date_to_percents[date])
        date_to_avg_percent[date] = avg_percent

    print(date_to_avg_percent)

    s = sum(date_to_avg_percent.values())
    print(s / len(date_to_avg_percent.values()))

    print("------------------------------------------------------------------------")
    print("Document Count: ", doc_count)
    print("Average Redaction Count: ", int(total_redaction_count / doc_count))
    print("Average Percent of Text Redacted: ", total_percent_text_redacted / doc_count)
    print("Average Number of Words Redacted: ", int(total_num_words_redacted / doc_count))

    # --------- FREQUENCIES OF PERCENT TEXT REDACTED PLOT ----
    plot2_x = list(percent_to_freq.keys())
    plot2_y = list(percent_to_freq.values())
    plt.bar(plot2_x, plot2_y, width=.8, color='#FF99AC')
    plt.title("Frequencies of Percent Text Redacted (Per Document)")
    plt.xlabel("Percent Text Redacted")
    plt.ylabel("Frequency")
    plt.show(block=True)

    # --------- PERCENT TEXT REDACTED OVER TIME PLOT ----
    plot3_x = list(date_to_avg_percent.keys())
    plot3_y = list(date_to_avg_percent.values())
    plt.bar(plot3_x, plot3_y, color='#FF99AC')
    plt.title("Avg Percent Text Redacted Per Document Per Year")
    plt.xlabel("Year")
    plt.ylabel("Avg Percent Text Redacted")
    plt.show(block=True)

def image_processing(jpg_file, doc_type):
    """ Returns:
        1) Redaction Count
        2) Redacted Text Area
        3) Estimated Number of Words Redacted
        for a single PAGE """

    import cv2
    import numpy as np
    img = cv2.imread(jpg_file)

    # ======= EDITING / FILTERING THE IMAGE =======
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.erode(gray, kernel, iterations=1)
    blur = cv2.GaussianBlur(img_erosion,(5,5),0)
    edited_img = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)

    # ======= Finding contours within the edited_img =======
    contours = cv2.findContours(edited_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cv2.drawContours(edited_img, contours, -1, (0,0,0), 3)

    # Identifying the Shape
    redactions = []
    next_potential = []

    if doc_type == "pdb":
        (potential, text_potential, type1, type2, type3) = cib_get_redaction_shapes_text_shapes(contours, edited_img)
    if doc_type == "cib":
        (potential, text_potential, type1, type2, type3) = cib_get_redaction_shapes_text_shapes(contours, edited_img)

    final_redactions = get_intersection_over_union(potential)
    redaction_count = len(final_redactions)
    [redacted_text_area, estimated_text_area, estimated_num_words_redacted] = get_stats(final_redactions, text_potential)

    return [redaction_count, redacted_text_area, estimated_text_area, estimated_num_words_redacted, potential, text_potential, type1, type2, type3]
