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
    # pdf_dir looks like "/Users/miabramel/Downloads/pdbs"
    os.chdir(pdf_dir)
    full_name = pdf_dir + "/" + pdf_file
    pages = convert_from_path(full_name, 300)
    full_name = full_name[:-4]
    for page in pages:
        page.save("%s-page%d.jpg" % (full_name, pages.index(page)), "JPEG")

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

def get_pdb_stats(redaction_shapes, text_shapes):
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
    estimated_text_area = text_area + redacted_text_area
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
            if iou > 0:
                rejects.append(potential[boxA])
                rejects.append(potential[boxB])

    # If the most common "redaction" is overlapping with 20 others,
    # assume it is a map
    if len(rejects) > 0 and Counter(rejects).most_common(1)[0][1] > 20:
        is_map = True
        # map_area.append(Counter(rejects).most_common(1)[0][0])

    if is_map:
        final_redactions = []
    else:
        final_redactions = [x for x in potential if x not in rejects]

    return final_redactions

def getIOU(boxA, boxB):
    """
    Determines the score of whether two redactions are overlapping.
    The following code is from
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-   detection/
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and divide it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

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
                    non_zero = np.count_nonzero(c)

                    # Determine that contour is 96% white space
                    if (non_zero / c.size) > .96:
                        # Append to potential list if redaction meets criteria
                        # If the redaction is oddly shaped
                        if w >= 7 and h >= 7 and  x != 0 and y != 0:
                            shape = x, x+w, y, y+h
                            potential.append(shape)
                        # If the redaction is a perfect rectangle
                        elif len(approx) == 4:
                            shape = x, x+w, y, y+h
                            potential.append(shape)

                # Detecting the text
                if peri > 25 and peri < 150:
                    approx = cv2.approxPolyDP(c, 0.04*peri, True)
                    (x, y, w, h) = cv2.boundingRect(approx)
                    shape = x, x+w, y, y+h
                    text_potential.append(shape)

    return (potential, text_potential)

def analyze_pdb_results(output_file):
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline, BSpline

    pdb_count = 0
    total_redaction_count = 0
    total_percent_text_redacted = 0
    total_num_words_redacted = 0

    # I will assume the max number of redactions in a PDB is 700.
    redaction_num_to_freq = {i:0 for i in range(700)}

    # I know the percents will go from 0 to 100, I make the steps 0.5
    percent_range = [percent*(0.5) for percent in range(200)]
    percent_range.append(100)
    percent_to_freq = {j:0 for j in percent_range}

    list_of_redaction_nums = []

    num_words_redacted_x = []
    num_redactions_y = []

    with open(output_file, mode='r') as output:
        reader = csv.reader(output)
        for row in reader:

            redaction_num = int(row[0])
            num_redactions_y.append(redaction_num)
            percent_text_redacted = float(row[1])
            num_words_redacted = int(row[2])
            num_words_redacted_x.append(num_words_redacted)
            list_of_redaction_nums.append(num_words_redacted)

            redaction_num_to_freq[50 * round(redaction_num / 50)] += 1
            # rounds the percent to the nearest .5
            percent_to_freq[round(percent_text_redacted * 200) / 2] += 1

            total_redaction_count += redaction_num
            total_percent_text_redacted += percent_text_redacted
            total_num_words_redacted += num_words_redacted

            pdb_count += 1

    # Making a list of 1, 2, 3, ... pdb_count
    num_words_keys = [i for i in range(0, pdb_count+1)]
    list_of_redaction_nums.sort()
    # Making a dictionary to be used for a graph
    num_words_dict = {num1:num2 for (num1, num2) in zip(num_words_keys, list_of_redaction_nums)}

    output.close()

    print()
    print("PDB Count: ", pdb_count)
    print("Average Redaction Count: ", int(total_redaction_count / pdb_count))
    print("Average Percent of Text Redacted: ", total_percent_text_redacted / pdb_count)
    print("Average Number of Words Redacted: ", int(total_num_words_redacted / pdb_count))

    # --------- FREQUENCIES OF PERCENT TEXT REDACTED PLOT ----
    plot2_x = list(percent_to_freq.keys())
    plot2_y = list(percent_to_freq.values())
    plt.bar(plot2_x, plot2_y, width=.8, color='#FF99AC')
    plt.title("Frequencies of Percent Text Redacted (Per PDB)")
    plt.xlabel("Percent Text Redacted")
    plt.ylabel("Frequency")
    plt.show(block=True)

    # ---------- FREQUENCIES OF REDACTIONS PLOT -------------
    plot1_x = list(redaction_num_to_freq.keys())
    plot1_y = list(redaction_num_to_freq.values())
    plt.bar(plot1_x, plot1_y, width=30, color='#FF99AC')
    plt.title("Frequencies of Redactions (Per PDB)")
    plt.xlabel("Number of Redactions")
    plt.ylabel("Frequency")
    plt.show(block=True)

    # ---------- NUMBER OF WORDS REDACTED PLOT --------------
    plot2_x = list(num_words_dict.keys())
    plot2_y = list(num_words_dict.values())
    xnew = np.linspace(min(plot2_x), max(plot2_x), 300)
    spl = make_interp_spline(plot2_x, plot2_y, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(xnew, power_smooth, color='#FF99AC', linewidth=4)
    plt.title("Number of Words Redacted (Per PDB)")
    plt.xlabel("PDB #")
    plt.ylabel("Number of Words Redacted")
    plt.show(block=True)

    # ---------- NUMBER OF REDACTIONS VS. NUMBER OF WORDS REDACTED PLOT --------------
    plot2_x = num_words_redacted_x
    plot2_y = num_redactions_y
    xnew = np.linspace(min(plot2_x), max(plot2_x), 300)
    spl = make_interp_spline(plot2_x, plot2_y, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(xnew, power_smooth, color='#FF99AC', linewidth=4)
    plt.title("Number of Redactions vs. Number of Words Redacted (Per PDB)")
    plt.xlabel("Number of Words Redacted")
    plt.ylabel("Number of Redactions")
    plt.show(block=True)

def image_processing(jpg_file):
    """ Returns:
        1) Redaction Count
        2) Redacted Text Area
        3) Estimated Number of Words Redacted
        for A SINGLE JPG PAGE OF A PDB."""

    import cv2
    import numpy as np
    img = cv2.imread(jpg_file)
    img_original = cv2.imread(jpg_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.erode(gray, kernel, iterations=1)
    blur = cv2.GaussianBlur(img_erosion,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)

    # Find contours and detect shape
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Identifying the Shape
    redactions = []
    next_potential = []

    (potential, text_potential) = get_redaction_shapes_text_shapes(contours)
    final_redactions = get_intersection_over_union(potential)
    redaction_count = len(final_redactions)
    [redacted_text_area, estimated_text_area, estimated_num_words_redacted] = get_pdb_stats(final_redactions, text_potential)

    return [redaction_count, redacted_text_area, estimated_text_area, estimated_num_words_redacted]
