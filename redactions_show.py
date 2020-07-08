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

    print("Redaction Count: ", len(final_redactions))
    print("Percent of Text Redacted: ", redaction_module.getPercentRedacted(final_redactions, text_potential))

    redaction_module.putRedactions(final_redactions, img)
    cv2.imshow("Image", img)

    """
    TODO: Will be fixed once we get the map detection figured out!

    # If there are more than 24 redactions, then I assume it's a map.
    if len(final_redactions) < 24:
        cv2.imshow("Image", img)
    else:
        cv2.imshow("Map", img_original)

    slash = pdf_file.rindex("/")
    underscore = pdf_file.rindex("-")
    period = pdf_file.rindex(".")
    docid = pdf_file[slash+1:underscore]
    pagenum = pdf_file[underscore+5:period]

    if len(final_redactions) != 0:
        for r in final_redactions:
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
    # redaction_module.take_screenshot(pdf_file)
    return ret

image_processing('/Users/miabramel/Downloads/pdbs/DOC_0005958912-page2.jpg')
