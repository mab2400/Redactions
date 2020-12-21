import redaction_module
import cv2
import numpy as np
import sys

def show_individual_page(jpg_file, doc_type):
    """ Analyzes the redactions within a single page of a PDB, displays the image with the redactions indicated. """

    [redaction_count, redacted_text_area, estimated_text_area, estimated_num_words_redacted, potential, text_potential, type1, type2, type3] = redaction_module.image_processing(jpg_file, doc_type)
    final_redactions = redaction_module.get_intersection_over_union(potential)
    final_type1 = redaction_module.get_intersection_over_union(type1)
    final_type2 = redaction_module.get_intersection_over_union(type2)
    final_type3 = redaction_module.get_intersection_over_union(type3)

    img = cv2.imread(jpg_file)
    redaction_module.drawRedactionRectangles(final_redactions, img)
    redaction_module.putRedactions(final_redactions, img)
    #redaction_module.put_type1_redactions(final_type1, img)
    #redaction_module.put_type2_redactions(final_type2, img)
    #redaction_module.put_type3_redactions(final_type3, img)

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

doc_type = sys.argv[1]
jpg_file = sys.argv[2]
show_individual_page(jpg_file, doc_type)
