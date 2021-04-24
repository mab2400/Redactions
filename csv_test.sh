#!/bin/sh

# $1: url2003
# $2: url2019

wget -O versions/first.pdf $1
wget -O versions/second.pdf $2

python3 stats.py cib batch ~/Redactions/versions ~/Redactions/versions_to
# Writes 2 rows to versions/output.csv

# Pull the estimated number of words redacted
python3 pull_data_from_csv.py versions/output.csv > num_words.csv
# Writes 1 row with 2 values: num_words2003,num_words2019

python3 count_num_pages.py versions_to/first.pdf versions_to/second.pdf > num_pages.csv
# Writes 1 row with 2 values: num_pages2003,num_pages2019

rm versions/output.csv
rm versions_to/first.pdf
rm versions_to/second.pdf
