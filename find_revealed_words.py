import csv
import sys
import re, string, timeit
from collections import Counter
import matplotlib.pyplot as plt

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
    exclude = set(string.punctuation)
    cleaned = ''.join(ch for ch in list_of_strings if ch not in exclude).lower()
    cleaned = cleaned.split()
    for word in cleaned:
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
                if country[0] == word or country[0] in word:
                    country_matches.append(country[0])
    return country_matches

def find_country_frequencies(all_documents):
    """Returns a dictionary of frequencies within all documents from 2003 or 2019"""
    return Counter(all_documents)

with open(sys.argv[1]) as csv_file:
    csv_reader = csv.reader(csv_file)
    row_count = 0

    all_countries_2003 = []
    all_countries_2019 = []

    print("Finding countries ...")

    for row in csv_reader:
        row_count += 1

        ocr_2019 = row[4]
        ocr_2003 = row[7]

        # Clean up
        ocr_2003 = clean_up(ocr_2003)
        ocr_2019 = clean_up(ocr_2019)

        # Find countries
        countries_2019 = find_countries(ocr_2019)
        countries_2003 = find_countries(ocr_2003)

        # Add to lists of ALL countries
        for country in countries_2003:
            all_countries_2003.append(country)
        for country in countries_2019:
            all_countries_2019.append(country)

        #print()
        #print(countries_2003)
        #print(countries_2019)
        #revealed_countries = revealed_words(countries_2019, countries_2003)

        # Find words in general
        #revealed_words_general = revealed_words(ocr_2019, ocr_2003)
        #print(revealed_countries)
        #print(revealed_words_general)


    print("FOUND COUNTRIES")
    dict1 = find_country_frequencies(all_countries_2003)
    dict1_list = sorted(dict1.items(), key=lambda x: x[1], reverse=True)
    countries1 = [s[0] for s in dict1_list]
    freqs1  = [s[1] for s in dict1_list]

    # Take the top 10
    countries1 = countries1[0:10]
    freqs1 = freqs1[0:10]

    plt.bar(range(10), freqs1, tick_label=countries1)
    plt.xlabel("Countries")
    plt.ylabel("Number of Mentions")
    plt.title("Top 10 Countries Mentioned in CIBs Released in 2003")
    plt.show()

    dict2 = find_country_frequencies(all_countries_2019)
    dict2_list = sorted(dict2.items(), key=lambda x: x[1], reverse=True)
    countries2  = [s[0] for s in dict2_list]
    freqs2 = [s[1] for s in dict2_list]
