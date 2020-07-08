# Redactions 

A Python-based project that utilizes OpenCV to detect redactions in declassified President's Daily Briefs from the mid-20th Century.

## redaction_module.py 

pdf_to_jpg: Converts a multiple-page PDF into multiple single JPG files. 
putRedactions: Writes the word "REDACTION" on top of the image in the locations where a redaction was detected.
drawTextRectangles: Draws the bounding rectangles of the detected text on the page.
getPercentRedacted: Calculates the percent of text redacted in a single PDB JPG file.
take_screenshot: Once the image is displayed in a window, press any key to take a screenshot and save to current directory.
get_non_overlapping_shapes: (Work-In-Progress) Returns a list of the non-overlapping redaction shapes.
get_redaction_shapes_text_shapes: Returns two lists, one of redaction shapes, another of text shapes. 
analyze_results: After running on a large batch of PDBs and outputting to a csv, calculate the stats. 

