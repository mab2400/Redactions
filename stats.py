import redaction_module
import os
import sys
import shutil
import csv
from tabulate import tabulate
import time

def analyze(directory, pdf_file, doc_type):
    """ Loops through each page within a single PDB and sums up the stats of each page to arrive at the overall total """

    total_redaction_count = 0
    total_redacted_text_area = 0
    total_estimated_text_area = 0
    total_estimated_num_words_redacted = 0

    # Split the pdb (which is a pdf file) into individual jpgs.
    redaction_module.pdf_to_jpg(directory, pdf_file)

    os.chdir(directory)
    for jpg_file in os.listdir(directory):
        # Iterating through each page of the PDB
        if jpg_file.endswith(".jpg"):

            [redaction_count, redacted_text_area, estimated_text_area, estimated_num_words_redacted, potential, text_potential] = redaction_module.image_processing(jpg_file, doc_type)

            total_redaction_count += redaction_count
            total_redacted_text_area += redacted_text_area
            total_estimated_text_area += estimated_text_area
            total_estimated_num_words_redacted += estimated_num_words_redacted

            # Crucial clean-up of jpg files (Note: If files are not removed, code will NOT work properly).
            os.remove(jpg_file)

    # Now that we've gone through each page, we need to calculate the stats for the document.
    if total_estimated_text_area != 0:
        total_percent_text_redacted = float(total_redacted_text_area / total_estimated_text_area)
    else:
        total_percent_text_redacted = 0

    data = []
    # open csv file and write the stats in a single row representing the document.
    with open('../pdb_output.csv', mode='a+') as output:
        output_writer = csv.writer(output, delimiter=',')
        row = [pdf_file, total_redaction_count, total_percent_text_redacted, total_estimated_num_words_redacted]
        data.append(row)
        print(tabulate(data, headers=["                  ", "                 ", "                     ", "                 ", "             "]))
        output_writer.writerow(row)
    output.close()

def test_batch(from_dir, to_dir, doc_type):
    """Iterates through all the PDBS (pdf files) in the given from directory, and moves them to the to directory when they are finished."""
    # NOTE: Both from_dir and to_dir MUST end with a SLASH.

    os.chdir(from_dir)
    for pdf_file in os.listdir(from_dir):
        if pdf_file.endswith(".pdf"):
            # Appends a row to the csv file "../pdb_output.csv" with the stats from that particular document
            analyze(from_dir, pdf_file, doc_type)

            # Moving to the 'to' directory since we're done analyzing it.
            destination = to_dir + pdf_file
            shutil.move(from_dir+ pdf_file, destination)

doc_type = sys.argv[1]
command = sys.argv[2]
from_dir = sys.argv[3]
to_dir = sys.argv[4]
# TODO: Make one of the arguments called doc_type
if command == "batch":
    print("File Name             Redaction Count      Percent Text Redacted    Num Words Redacted")
    test_batch(from_dir, to_dir, doc_type)
elif command == "analyze":
    redaction_module.analyze_results("../pdb_output.csv")
