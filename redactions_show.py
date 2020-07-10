import redaction_module

def image_processing(pdf_file):
    import cv2
    import numpy as np
    img = cv2.imread(pdf_file)
    img_original = cv2.imread(pdf_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.erode(gray, kernel, iterations=1)
    blur = cv2.GaussianBlur(img_erosion,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)

    # Find contours and detect shape
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    map_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    redactions = []
    next_potential = []

    (potential, text_potential) = redaction_module.get_redaction_shapes_text_shapes(contours)
    (final_redactions, is_map) = redaction_module.get_intersection_over_union(potential)
    stats = redaction_module.get_stats(final_redactions, text_potential)

    if not is_map:
        redaction_module.putRedactions(final_redactions, img)
        print("Redaction Count: ", len(final_redactions))
        print("Percent of Text Redacted: ", stats[0])
        print("Estimated Number of Words Redacted: ", stats[1])
    else:
        print("Map: True")

    cv2.imshow("Image", img)
    cv2.waitKey()
    redaction_module.take_screenshot(pdf_file)

image_processing('/Users/miabramel/Downloads/pdbs/DOC_0005958911-page7.jpg')
