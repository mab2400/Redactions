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

def contains_number(string):
    for char in string:
        if char.isnumeric():
            return True
    return False

def clean_up(list_of_strings):
    """Only takes the words without numbers and with all ASCII chars"""
    new_list = []
    for word in list_of_strings:
        if contains_number(word) == False and word.isascii() == True:
            new_list.append(word)
    return new_list

def find_countries(list_of_strings):
    """Returns a list of countries that appear in the OCR"""
    country_matches = []
    with open("countries.txt", 'r') as csv_file:
        csv_read = csv.reader(csv_file)
        for country in csv_read:
            for word in list_of_strings:
                if country[0] == word:
                    country_matches.append(country[0])
    return country_matches

with open(sys.argv[1]) as csv_file:
    csv_reader = csv.reader(csv_file)
    row_count = 0
    for row in csv_reader:
        row_count += 1

        ocr_2019 = row[4]
        ocr_2003 = row[7]

        # Clean up
        exclude = set(string.punctuation)
        ocr_2019 = ''.join(ch for ch in ocr_2019 if ch not in exclude).lower()
        ocr_2003 = ''.join(ch for ch in ocr_2003 if ch not in exclude).lower()
        ocr_2019 = clean_up(ocr_2019.split())
        ocr_2003 = clean_up(ocr_2003.split())

        # Find countries
        countries_2019 = find_countries(ocr_2019)
        countries_2003 = find_countries(ocr_2003)
        revealed_countries = revealed_words(countries_2019, countries_2003)

        # Find words in general
        revealed_words_general = revealed_words(ocr_2019, ocr_2003)

        print()
        print(revealed_countries)
        print(revealed_words_general)



