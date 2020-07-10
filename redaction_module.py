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

def get_stats(redaction_shapes, text_shapes):
    """ Calculates the percent of text redacted, estimated number of words redacted. """
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
    # Now, I want to divide redacted_text_area by the (estimated) area of a single word in order to estimate
    # the number of words redacted.
    estimated_word_area = 4665
    estimated_num_words = int(redacted_text_area / estimated_word_area)
    if estimated_text_area == 0:
        return (0, estimated_num_words)
    else:
        return ((redacted_text_area / estimated_text_area), estimated_num_words)

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

def get_intersection_over_union(potential):
    import cv2
    # Input is a list of shapes, output is (final_redactions, map_area)
    """ Returns non-overlapping redactions, and if necessary the map area. """

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

    return final_redactions, is_map

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

def get_non_overlapping_shapesOLD(next_potential):
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

def analyze_results(output_file):
    import matplotlib.pyplot as plt

    map_count = 0
    total_redaction_count_with_zeros = 0
    total_percent_redacted_with_zeros = 0
    total_words_redacted_with_zeros = 0
    pdb_count_with_zeros = 0
    total_redaction_count_without_zeros = 0
    total_percent_redacted_without_zeros = 0
    total_words_redacted_without_zeros = 0
    pdb_count_without_zeros = 0

    redaction_num_to_freq_with_zeros = {i:0 for i in range(100)}
    # I know the percents will go from 0 to 100, I make the steps 0.5
    percent_range = [percent*(0.5) for percent in range(200)]
    percent_range.append(100)
    percent_to_freq_with_zeros = {j:0 for j in percent_range}
    # I'll assume the max number of words that could be redacted on a single page is 500.
    num_words_to_freq_with_zeros = {k:0 for k in range(500)}

    redaction_num_to_freq_without_zeros = {i:0 for i in range(100)}
    percent_to_freq_without_zeros = {j:0 for j in percent_range}
    # I'll assume the max number of words that could be redacted on a single page is 500.
    num_words_to_freq_without_zeros = {k:0 for k in range(500)}

    with open(output_file, mode='r') as output:
        reader = csv.reader(output)
        for row in reader:

            redaction_num = int(row[0])
            percent_text_redacted = float(row[1])
            num_words_redacted = int(row[2])

            if int(row[3]) == 0:
                # NOT A MAP

                redaction_num_to_freq_with_zeros[redaction_num] += 1
                # rounds the percent to the nearest .5
                percent_to_freq_with_zeros[round(percent_text_redacted * 200) / 2] += 1
                num_words_to_freq_with_zeros[num_words_redacted] += 1

                total_redaction_count_with_zeros += redaction_num
                total_percent_redacted_with_zeros += percent_text_redacted
                total_words_redacted_with_zeros += num_words_redacted

                pdb_count_with_zeros += 1

                if redaction_num > 0 and percent_text_redacted > 0 and num_words_redacted > 0:
                    redaction_num_to_freq_without_zeros[redaction_num] += 1
                    # rounds the percent to the nearest .5
                    percent_to_freq_without_zeros[round(percent_text_redacted * 200) / 2] += 1
                    num_words_to_freq_without_zeros[num_words_redacted] += 1

                    total_redaction_count_without_zeros += redaction_num
                    total_percent_redacted_without_zeros += percent_text_redacted
                    total_words_redacted_without_zeros += num_words_redacted

                    pdb_count_without_zeros += 1
            else:
                # Is a Map
                map_count += 1
                pdb_count_with_zeros += 1
                pdb_count_without_zeros += 1


    output.close()

    print()
    print("PDB Count with zeros: ", pdb_count_with_zeros)
    print("Average Redaction Count with zeros: ", int(total_redaction_count_with_zeros / pdb_count_with_zeros))
    print("Average Percent of Text Redacted with zeros: ", total_percent_redacted_with_zeros / pdb_count_with_zeros)
    print("Average Number of Words Redacted with zeros: ", int(total_words_redacted_with_zeros / pdb_count_with_zeros))
    print()
    print("PDB Count without zeros: ", pdb_count_without_zeros)
    print("Average Redaction Count without zeros: ", int(total_redaction_count_without_zeros / pdb_count_without_zeros))
    print("Average Percent of Text Redacted without zeros: ", total_percent_redacted_without_zeros / pdb_count_without_zeros)
    print("Average Number of Words Redacted without zeros: ", int(total_words_redacted_without_zeros / pdb_count_without_zeros))
    print("Map Count: ", map_count)

    # --------- FREQUENCIES OF PERCENT TEXT REDACTED PLOT ----
    # plot2_x = list(percent_to_freq.keys())
    # plot2_y = list(percent_to_freq.values())
    # plt.scatter(plot2_x, plot2_y, color='#FF99AC', linewidth=4)
    # plt.title("Frequencies of Percent Text Redacted")
    # plt.xlabel("Percent Text Redacted on One Page")
    # plt.ylabel("Frequency")
    # plt.show(block=True)

    # ---------- FREQUENCIES OF REDACTIONS PLOT (WITH ZEROS) -------------
    plot1_x = list(redaction_num_to_freq_with_zeros.keys())
    plot1_y = list(redaction_num_to_freq_with_zeros.values())
    plt.plot(plot1_x, plot1_y, color='#FF99AC', linewidth=4)
    plt.title("Frequencies of Redactions Per Page (with zeros)")
    plt.xlabel("Number of Redactions")
    plt.ylabel("Frequency")
    plt.show(block=True)

    # ---------- FREQUENCIES OF NUMBER OF WORDS REDACTED PLOT (WITH ZEROS) --------------
    plot2_x = list(num_words_to_freq_with_zeros.keys())
    plot2_y = list(num_words_to_freq_with_zeros.values())
    plt.scatter(plot2_x, plot2_y, color='#FF99AC', linewidth=4)
    plt.title("Frequencies of Number of Words Redacted Per Page (with zeros)")
    plt.xlabel("Number of Words Redacted")
    plt.ylabel("Frequency")
    plt.show(block=True)

    # ---------- FREQUENCIES OF REDACTIONS PLOT (WITHOUT ZEROS) --------------
    plot3_x = list(redaction_num_to_freq_without_zeros.keys())
    plot3_y = list(redaction_num_to_freq_without_zeros.values())
    plt.plot(plot3_x, plot3_y, color='#FF99AC', linewidth=4)
    plt.title("Frequencies of Redactions Per Page (without zeros)")
    plt.xlabel("Number of Redactions")
    plt.ylabel("Frequency")
    plt.show(block=True)

    # ---------- FREQUENCIES OF NUMBER OF WORDS REDACTED PLOT (WITHOUT ZEROS) --------------
    plot4_x = list(num_words_to_freq_without_zeros.keys())
    plot4_y = list(num_words_to_freq_without_zeros.values())
    plt.scatter(plot4_x, plot4_y, color='#FF99AC', linewidth=4)
    plt.title("Frequencies of Number of Words Redacted Per Page (without zeros)")
    plt.xlabel("Number of Words Redacted")
    plt.ylabel("Frequency")
    plt.show(block=True)
