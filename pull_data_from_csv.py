import csv
import sys
import re, string, timeit


"""
Reads from the csv file versions/output.csv which has two lines.
The first line is for version #1 (2003) results.
The second line is for version #2 (2019) results.
"""

with open(sys.argv[1]) as csv_file:
    csv_reader = csv.reader(csv_file)
    print("%s,%s" % (csv_reader[0][3], csv_reader[1][3]))

