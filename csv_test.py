import csv
import sys
import os

def get_num_words_and_pages(url2003, url2019):
    pid = os.fork()
    if pid == 0:
        os.execl("./csv_test.sh", "csv_test.sh", url2003, url2019)
    if pid > 0:
        os.waitpid(pid, 0)
    # The two num_words values have been written into num_words.csv.
    # The two page counts have been written into num_pages.csv.

def get_pages_exempt_2003(OCR):
    """ Finds occurrences of the message Next 1 Page(s) In Document Exempt,
        and determines the number of missing / fully redacted pages """
    num_pages_exempt = 0
    for i,word in enumerate(OCR):
        if word == "Exempt":
            num_pages_exempt += int(OCR[i-4])
    return num_pages_exempt

def build_csv():
    """

    each row looks like:
    [cibid, url2003, year2003, num_words_redacted2003, num_pages2003, num_pages_exempt2003, url2019, year2019, num_words_redacted2019, num_pages2019]

    """
    with open('cibcia_body.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        with open('cib_pairs.csv', mode='w') as pairs_file:
            csv_writer = csv.writer(pairs_file, delimiter=',')
            row_num = 0
            for row in csv_reader:
                if row_num == 0:
                    row_num += 1
                    continue
                pairs_row = []
                url2003 = ""
                url2019 = ""
                pairs_row.append(row[1]) # cib ID ================= pairs_row[0]

                # ==== 2003 INFO ====

                if len(row[8]) != 0:
                    url2003 = "https://www.cia.gov/readingroom/docs/{}.pdf".format(row[8][41:])
                pairs_row.append(url2003) # URL 2003 =============== pairs_row[1]
                pairs_row.append('2003') # year 2003 =============== pairs_row[2]
                pairs_row.append("num_words_redacted2003_PLACEHOLDER") # for num_words_redacted 2003 ========== pairs_row[3]
                pairs_row.append("num_pages2003_PLACEHOLDER") # ================== pairs_row[4]
                pairs_row.append(get_pages_exempt_2003(row[7].split())) # Finding the number of pages exempt in the 2003 version ========== pairs_row[5]

                # ==== 2019 INFO ====

                url2019 = row[3]
                pairs_row.append(url2019) # URL 2019 ============== pairs_row[6]
                with open('cib_meta.csv') as csv_meta:
                    csv_reader_meta = csv.reader(csv_meta)
                    for row1 in csv_reader_meta:
                        if row1[4][1:] == row[1]: # finding the matching cibid
                            pairs_row.append(row1[9][-4:]) # year 2019-ish ============= pairs_row[7]
                            break
                pairs_row.append("num_words_redacted2019_PLACEHOLDER") # for num_words_redacted 2019 =========== pairs_row[8]
                pairs_row.append("num_pages2019_PLACEHOLDER") # =================== pairs_row[9]


                # Filling in placeholders

                get_num_words_and_pages(url2003, url2019)

                with open('num_words.csv') as num_words:
                    """ num_words2003,num_words2019  """
                    num_words_reader = csv.reader(num_words)
                    for r in num_words_reader:
                        pairs_row[3] = r[0] # num_words2003
                        pairs_row[8] = r[1] # num_words2019
                with open('num_pages.csv') as num_pages_total:
                    """ num_pages2003,num_pages2019  """
                    num_pages_reader = csv.reader(num_pages_total)
                    for r in num_pages_reader:
                        pairs_row[4] = r[0] # num_pages2003
                        pairs_row[9] = r[1] # num_pages2019

                """
                # If any items are empty, don't add that row to the file.
                any_empty = False
                for item in pairs_row:
                    if len(item) == 0:
                        any_empty = True
                if any_empty:
                    continue
                else:
                """

                csv_writer.writerow(pairs_row)
                pairs_file.flush()

build_csv()



