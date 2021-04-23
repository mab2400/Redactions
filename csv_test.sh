#!/bin/sh

# Takes $1 (the URL of the first document) and $2 (the URL of the second document)

wget -O versions/first.pdf $1
wget -O versions/second.pdf $2

python3 stats.py cib batch ~/Redactions/versions ~/Redactions/versions_to
# The stats.py program writes to versions/output.csv

# Pull the estimated number of words redacted
python3 pull_data_from_csv.py versions/output.csv > num_words.csv

rm versions/output.csv
rm versions_to/first.pdf
rm versions_to/second.pdf
