import redaction_module
import os
import csv

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

            [redaction_count, redacted_text_area, estimated_text_area, estimated_num_words_redacted] = redaction_module.image_processing(jpg_file)

            total_redaction_count += redaction_count
            total_redacted_text_area += redacted_text_area
            total_estimated_text_area += estimated_text_area
            total_estimated_num_words_redacted += estimated_num_words_redacted

    # Now that we've gone through each page, we need to calculate the stats for the PDB.
    if total_estimated_text_area != 0:
        total_percent_text_redacted = float(total_redacted_text_area / total_estimated_text_area)
    else:
        total_percent_text_redacted = 0

    # open csv file and write the stats in a single row representing the pdb.
    with open('/Users/miabramel/Desktop/Redactions/pdb_output.csv', mode='a') as output:
        output_writer = csv.writer(output, delimiter=',')
        row = [total_redaction_count, total_percent_text_redacted, total_estimated_num_words_redacted]
        print(row)
        output_writer.writerow(row)
    output.close()

def test_batch(pdb_directory):
    """Iterates through all the PDBS (pdf files) in the given directory"""
    os.chdir(pdb_directory)
    for pdf_file in os.listdir(pdb_directory):
        if pdf_file.endswith(".pdf"):
            # Appends a row to the csv file "pdb_output.csv" with the stats from that particular PDB
            analyze_pdb(pdb_directory, pdf_file)

test_batch("/Users/miabramel/Downloads/newpdbs")
redaction_module.analyze_pdb_results("/Users/miabramel/Desktop/Redactions/pdb_output.csv")
