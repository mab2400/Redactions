import redaction_module
import os
import csv

def image_processing(jpg_file):
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

    (potential, text_potential) = redaction_module.get_redaction_shapes_text_shapes(contours)
    final_redactions = redaction_module.get_intersection_over_union(potential)
    redaction_count = len(final_redactions)
    [redacted_text_area, estimated_text_area, estimated_num_words_redacted] = redaction_module.get_pdb_stats(final_redactions, text_potential)

    return [redaction_count, redacted_text_area, estimated_text_area, estimated_num_words_redacted]

def analyze_pdb(pdb_directory, pdf_file):
    """ Loops through each page within a single PDB and sums up the stats of each page to arrive at the overall total """

    total_redaction_count = 0
    total_redacted_text_area = 0
    total_estimated_text_area = 0
    total_estimated_num_words_redacted = 0

    # Split the pdb (which is a pdf file) into individual jpgs.
    redaction_module.pdf_to_jpg(pdb_directory, pdf_file)

    os.chdir(pdb_directory)
    for jpg_file in os.listdir(pdb_directory):
        # Iterating through each page of the PDB
        if jpg_file.endswith(".jpg"):

            [redaction_count, redacted_text_area, estimated_text_area, estimated_num_words_redacted] = image_processing(jpg_file)

            total_redaction_count += redaction_count
            total_redacted_text_area += redacted_text_area
            total_estimated_text_area += estimated_text_area
            total_estimated_num_words_redacted += estimated_num_words_redacted

    # Now that we've gone through each page, we need to calculate the stats for the PDB.
    total_percent_text_redacted = total_redacted_text_area / total_estimated_text_area

    # open csv file and write the stats in a single row representing the pdb.
    with open('/Users/miabramel/Desktop/Redactions/pdb_output.csv', mode='a') as output:
        output_writer = csv.writer(output, delimiter=',')
        row = [total_redaction_count, total_percent_text_redacted, total_estimated_num_words_redacted]
        print(row)
        output_writer.writerow(row)
    output.close()

def test_batch(pdb_directory):
    """Iterates through all the PDBS (pdf files) in the given directory"""
    os.chdir(pdb_directory)
    for pdf_file in os.listdir(pdb_directory):
        if pdf_file.endswith(".pdf"):
            # Appends a row to the csv file "pdb_output.csv" with the stats from that particular PDB
            analyze_pdb(pdb_directory, pdf_file)

# test_batch("/Users/miabramel/Downloads/newpdbs")
redaction_module.analyze_pdb_results("/Users/miabramel/Desktop/Redactions/pdb_output.csv")
