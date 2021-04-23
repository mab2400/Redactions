import csv
import sys
import re, string, timeit


def revealed_words(ocr_later, ocr_earlier):
    """
    ocr_earlier: (XXX)vietnam, (XXX)france, (XXX)germany
    ocr_later: (XXX)vietnam, vietnam, (XXX)france, (XXX)germany, china, china

    """

    for word in ocr_earlier:
        if word in ocr_later:
            ocr_later.remove(word)
    return ocr_later


def num_exempt(list_of_strings):
    """ Finds occurrences of the message Next 1 Page(s) In Document Exempt,
        and determines the number of missing / fully redacted pages """

    num_pages_exempt = 0
    for i,word in enumerate(list_of_strings):
        if word == "Exempt":
            num_pages_exempt += int(list_of_strings[i-4])
    return num_pages_exempt

with open(sys.argv[1]) as csv_file:
    csv_reader = csv.reader(csv_file)
    row_count = 0
    for row in csv_reader:
        row_count += 1

        ocr_2019 = row[4].split()
        ocr_2003 = row[7].split()

        # Skip over it if there's no OCR to begin with
        if len(ocr_2019) == 0 and len(ocr_2003) == 0:
            continue

        exempt_2003 = num_exempt(ocr_2003)
        exempt_2019 = num_exempt(ocr_2019)

        if exempt_2003 == 0 and exempt_2019 == 0:
            continue

        print("Num pages exempt in 2003: %d" % exempt_2003)
        print("Num pages exempt in 2019: %d" % exempt_2019)
        print()






