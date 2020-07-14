import redaction_module
import os

def image_processing(pdf_file):
    import cv2
    import numpy as np
    import csv
    import shutil
    img = cv2.imread(pdf_file)
    img_original = cv2.imread(pdf_file)
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
    (final_redactions, is_map) = redaction_module.get_intersection_over_union(potential)

    if is_map:
        print(pdf_file)
        shutil.copy2('/Users/miabramel/Downloads/pdbs/' + pdf_file, '/Users/miabramel/Downloads/mapexamples/' + pdf_file)

def test_batch(directory):
    # Iterate through all .jpg files in the given directory
    # redaction_module.pdf_to_jpg(directory)
    os.chdir(directory)
    for jpg_file in os.listdir(directory):
        if jpg_file.endswith(".jpg") and not jpg_file.endswith("screenshot.jpg"):
            image_processing(jpg_file)

# redaction_module.pdf_to_jpg("/Users/miabramel/Downloads/pdbs")
test_batch("/Users/miabramel/Downloads/pdbs")
