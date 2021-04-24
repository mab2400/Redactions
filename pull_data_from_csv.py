import csv
import sys
import re, string, timeit

"""
output.csv looks like:

second.pdf, XX, XX, XX
first.pdf, XX, XX, XX

(or order could be flipped)
"""

with open(sys.argv[1]) as csv_file:
    csv_reader = csv.reader(csv_file)
    val1 = 0
    val2 = 0
    for row in csv_reader:
        if row[0] == "first.pdf":
            val1 = row[3]
        if row[0] == "second.pdf":
            val2 = row[3]
    print("%s,%s" % (val1,val2))

