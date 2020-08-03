import redaction_module
import cv2
import numpy as np
import sys

def show_individual_page(jpg_file):
    """ Analyzes the redactions within a single page of a PDB, displays the image with the redactions indicated. """

    [redaction_count, redacted_text_area, estimated_text_area, estimated_num_words_redacted] = redaction_module.image_processing(jpg_file)

    img = cv2.imread(jpg_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.erode(gray, kernel, iterations=1)
    blur = cv2.GaussianBlur(img_erosion,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)

    # Find contours and detect shape
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cv2.drawContours(img, contours, -1, (0,255,0), 3)

    # Identifying the Shape
    (potential, text_potential) = redaction_module.get_redaction_shapes_text_shapes(contours)
    final_redactions = redaction_module.get_intersection_over_union(potential)
    # redaction_module.drawRedactionRectangles(final_redactions, img)
    # redaction_module.putRedactions(final_redactions, img)

    print("Redaction Count: ", redaction_count)
    if estimated_text_area != 0:
        percent_text_redacted = float(redacted_text_area / estimated_text_area)
    else:
        percent_text_redacted = 0
    print("Percent of Text Redacted: ", percent_text_redacted)
    print("Estimated Number of Words Redacted: ", estimated_num_words_redacted)

    cv2.imshow("Image", img)
    cv2.waitKey()
    redaction_module.take_screenshot(jpg_file)

jpg_file = sys.argv[1]
show_individual_page(jpg_file)
