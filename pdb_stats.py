import redaction_module
import os
import sys
import shutil
import csv
from tabulate import tabulate
import time

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

            [redaction_count, redacted_text_area, estimated_text_area, estimated_num_words_redacted, is_map] = redaction_module.image_processing(jpg_file)

            total_redaction_count += redaction_count
            total_redacted_text_area += redacted_text_area
            total_estimated_text_area += estimated_text_area
            total_estimated_num_words_redacted += estimated_num_words_redacted

            # Crucial clean-up of jpg files (Note: If files are not removed, code will NOT work properly).
            os.remove(jpg_file)

    # Now that we've gone through each page, we need to calculate the stats for the PDB.
    if total_estimated_text_area != 0:
        total_percent_text_redacted = float(total_redacted_text_area / total_estimated_text_area)
    else:
        total_percent_text_redacted = 0

    data = []
    # open csv file and write the stats in a single row representing the pdb.
    with open('/Users/carriehaykellar/History_Lab/Redaction Project/Redactions-master/pdb_output.csv', mode='a+') as output:
        output_writer = csv.writer(output, delimiter=',')
        row = [pdf_file, total_redaction_count, total_percent_text_redacted, total_estimated_num_words_redacted, is_map]
        data.append(row)
        print(tabulate(data, headers=["                  ", "                 ", "                     ", "                 ", "             "]))
        output_writer.writerow(row)
    output.close()

def test_batch(pdb_from_directory, pdb_to_directory):
    """Iterates through all the PDBS (pdf files) in the given from directory, and moves them to the to directory when they are finished."""
    # NOTE: Both pdb_from_directory and pdb_to_directory MUST end with a SLASH.

    os.chdir(pdb_from_directory)
    for pdf_file in os.listdir(pdb_from_directory):
        if pdf_file.endswith(".pdf"):
            # Appends a row to the csv file "pdb_output.csv" with the stats from that particular PDB
            analyze_pdb(pdb_from_directory, pdf_file)

            # Moving to the 'to' directory since we're done analyzing it.
            destination = pdb_to_directory + pdf_file
            shutil.move(pdb_from_directory + pdf_file, destination)

command = sys.argv[1]
if command == "batch":
    pdb_from_directory = sys.argv[2]
    pdb_to_directory = sys.argv[3]
    print("File Name             Redaction Count      Percent Text Redacted    Num Words Redacted    Map Present")
    test_batch(pdb_from_directory, pdb_to_directory)
elif command == "analyze":
    redaction_module.analyze_pdb_results("pdb_output.csv")

