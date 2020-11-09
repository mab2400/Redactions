# Redactions Project 

A Python-based project that utilizes OpenCV to detect and analyze redactions in declassified President's Daily Briefs and Central Intelligence Bulletins from the mid-20th Century.

![](images/beforeafter.jpg)

# Analyzing Redactions

## Getting Started

Prepare two directories:
- A directory containing _only_ pdf files of the documents (the "from" directory)
- An empty directory (the "to" directory)

As the script analyzes each document, it will move the files from the "from" directory to the "to" directory. This is a safety measure; if the script gets interrupted at any point, you can pick up where you left off and avoid reanalyzing any documents. 

## Running the Script

To find the redactions in a __batch__ of documents and generate a CSV file containing the data: 

```bash
python3 stats.py <doc_type> batch <from_dir> <to_directory>
```
- __doc_type__ indicates the document type of the files you inputted. For example, if your files are President's Daily Briefs, replace doc_type with pdb. If your files are Central Intelligence Bulletins, replace doc_type with cib. 
- You should expect to see a table being generated on the screen. Note that the raw data will be stored in __output.csv__ for future use if necessary.

To generate graphs and __analyze__ the data in the CSV file: 

```bash
python3 stats.py analyze
```
You should expect graphs to appear in a pop-up window. See below for an example.

<div align="center">
<img src="images/fullgraph1.png" width="500">
</div>

# Analyzing and Displaying a Single Page

Analyzing the redactions on a single page of a document file and display an image with the identified redactions.

## Running the Script

```bash
python3 redactions_show.py <jpg_filepath>
```
- __jpg_filepath__ is the filepath for a single page of a document. It must be a JPG file.
- Running this script will calculate the number of redactions, the percent of text on the page that was redacted, as well as the estimated number of words that were redacted. It will also open a pop-up window containing an image that identifies the locations of the redactions on the page.
- Once the pop-up window opens, press any key, which automatically takes a screenshot so you can save the analyzed image for future reference.

# Results

## NER Top 15 Entities - Redactions per Mention
China, Moscow, USSR more heavily redacted. Hanoi less redacted.
![](images/best1.png)

## Weighted Redactions per Year
![](images/best2.png)
# How Does It Work?

Our script makes use of OpenCV’s built-in contour and shape detection features to identify the white redaction boxes on each page of a PDB document. 

![](images/step1.png)
![](images/step2.png)
![](images/step3.png)

By calculating the area of the redaction in comparison to text on the page, we can estimate the number of words that were redacted as well as what percent of text on the page was redacted.

