import PyPDF2
import sys
from PyPDF2 import PdfFileReader

reader1 = PdfFileReader(open(sys.argv[1], 'rb'))
reader2 = PdfFileReader(open(sys.argv[2], 'rb'))
print("%s,%s" % (reader1.getNumPages(), reader2.getNumPages()))
