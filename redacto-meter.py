def image_processing(pdf_file):
    import cv2
    import imutils
    import numpy as np
    import math
    import time
    from collections import Counter

    # Erode and threshold image
    img = cv2.imread(pdf_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.erode(gray, kernel, iterations=1)
    blur = cv2.GaussianBlur(img_erosion,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)

    # Find contours and detect shape
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    map_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    ret = []
    redactions = []
    potential = []
    possible_c = []
    next_potential = []
    total_area = 0

    for c in contours:

        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            shape = 0
            peri = cv2.arcLength(c, True)
     
            # Smallest size of a redaction
            if peri > 700:
                # Compute the bounding box of the contour
                approx = cv2.approxPolyDP(c, 0.04*peri, True)
                (x, y, w, h) = cv2.boundingRect(approx)
                possible_c.append(c)
                non_zero = np.count_nonzero(c)

                #Determine that contour is 96% white space
                if (non_zero/c.size) > .96:

                    # Append to potential list if redaction meets criteria
                    if w>=10 and h>=10 and x != 0 and y != 0:
                        shape = x, y, w+x, h+y
                        potential.append(shape)

    # Call function to determine overlapping redactions and the map
    final_redactions, map_area = get_intersection_over_union(potential)

    parsed = []
    for boxA in final_redactions:
        cv2.rectangle(img, (int(boxA[0]), int(boxA[1])),(int(boxA[2]), int(boxA[3])),color=(0, 255, 0), thickness=1)   
        
    print("Redaction Count: ", len(final_redactions))
    print("Map Count:", len(map_area))

    cv2.imwrite('/Users/carriehaykellar/Desktop/Test_gray.jpg', img)
    cv2.waitKey()

    return ret


def get_intersection_over_union(potential):    
    """ 
    Returns non-overlapping redactions, and if necessary the map area.
    """
    from collections import Counter
    
    rejects = []
    map_area = []
    final_redactions = []

    # iterates through potential redaction list
    for boxA in range(0, len(potential)-1):
        for boxB in range(boxA+1, len(potential)):
            iou = getIOU(potential[boxA], potential[boxB])

            # if redactions overlap, append to reject list
            if iou > 0:
                rejects.append(potential[boxA], potential[boxB])    

    # if the most common "redaction" is overlapping with 20 others,
    # assume it is a map
    if Counter(rejects).most_common(1)[0][1] > 20:
        map_area.append(Counter(rejects).most_common(1)[0][0])


    final_redactions = [x for x in potential if x not in rejects] 

    return final_redactions, map_area

def getIOU(boxA, boxB):    
    """ 
    Determines the score of whether two redactions are overlapping.
    The following code is from 
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # print("Intersection", interArea)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    # print(iou)
    return iou
