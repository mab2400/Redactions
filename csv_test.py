import csv
import sys
import os

def get_num_words(url1, url2):
    pid = os.fork()
    if pid == 0:
        os.execl("./csv_test.sh", "csv_test.sh", url1, url2)
    if pid > 0:
        os.waitpid(pid, 0)
    # At this point, the two values have been written into num_words.csv.

def build_csv():
    """

    each row looks like:
    [cibid, url1, year1, num_words_redacted1, url2, year2, num_words_redacted2]

    TODO: Add num pages in 2003, num pages exempt in 2003, num pages in 2019

    """
    with open('cibcia_body.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        with open('cib_pairs.csv', mode='w') as pairs_file:
            csv_writer = csv.writer(pairs_file, delimiter=',')
            for row in csv_reader:
                pairs_row = []
                url1 = ""
                url2 = ""
                pairs_row.append(row[1]) # cib ID
                url1 = row[3]
                pairs_row.append(url1) # URL 1
                with open('cib_meta.csv') as csv_meta:
                    csv_reader_meta = csv.reader(csv_meta)
                    for row1 in csv_reader_meta:
                        if row1[4][1:] == row[1]:
                            pairs_row.append(row1[9][-4:]) # year 1
                            break
                pairs_row.append("placeholder") # for num_words_redacted 1




                if len(row[8]) != 0:
                    url2 = "https://www.cia.gov/readingroom/docs/{}.pdf".format(row[8][41:])
                pairs_row.append(url2) # URL 2
                pairs_row.append('2003') # year 2
                pairs_row.append("placeholder") # for num_words_redacted 2
                get_num_words(url1, url2)
                with open('num_words.csv') as num_words:
                    num_words_reader = csv.reader(num_words)
                    for r in num_words_reader:
                        pairs_row[3] = r[0]
                        pairs_row[6] = r[1]
                # If any items are empty, don't add that row to the file.
                any_empty = False
                for item in pairs_row:
                    if len(item) == 0:
                        any_empty = True
                if any_empty:
                    continue
                else:
                    csv_writer.writerow(pairs_row)
                    pairs_file.flush()

build_csv()



