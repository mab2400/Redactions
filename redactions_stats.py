import redaction_module
import os

def image_processing(pdf_file):
    import cv2
    import numpy as np
    import csv
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

    # ------------------------------------------------------------------------------------------------

    # STEPS:
    # 1) Find the shapes in the image (next_potential)
    # 2) Figure out which shapes are overlapping, remove them from the list of shapes (final_redactions)
    # 3) Put these non-overlapping shapes into a list
    # 4) Go through the non-overlapping shapes and call putText to write on the image

    # ------------------------------------------------------------------------------------------------

    # 1) Find the shapes in the image (next_potential)

    # Identifying the Shape
    ret = []
    redactions = []
    next_potential = []
    total_area = 0

    (potential, text_potential) = redaction_module.get_redaction_shapes_text_shapes(contours)

    for shape in potential:
        roi = thresh[shape[2]:shape[3], shape[0]:shape[1]]
        non_zero = np.count_nonzero(roi)
        # Maybe we should change > 0.95. Usually it's 0.3 or less.
        # if (non_zero/roi.size) > 0.95:
        next_potential.append(shape)

    final_redactions = redaction_module.get_non_overlapping_shapes(next_potential)

    # If there are more than 24 redactions, then I assume it's a map.
    redaction_count = len(final_redactions)
    if redaction_count < 24 and redaction_count > 0:
        percent_redacted = redaction_module.getPercentRedacted(final_redactions, text_potential)
        print()
        print(jpg_file)
        print("Redaction Count: ", redaction_count)
        print("Percent of Text Redacted: ", percent_redacted)
        with open('/Users/miabramel/Desktop/Redactions/output.csv', mode='a') as output:
            output_writer = csv.writer(output, delimiter=',')
            output_writer.writerow([redaction_count, percent_redacted])
        output.close()

    return ret

def test_batch(directory):
    # Iterate through all .jpg files in the given directory
    # redaction_module.pdf_to_jpg(directory)
    os.chdir(directory)
    for jpg_file in os.listdir(directory):
        if jpg_file.endswith(".jpg") and not jpg_file.endswith("screenshot.jpg"):
            image_processing(jpg_file)

# test_batch("/Users/miabramel/Downloads/pdbs")
redaction_module.analyze_results("/Users/miabramel/Desktop/Redactions/output.csv")
