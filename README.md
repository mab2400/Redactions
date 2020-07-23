# Redactions Project 

A Python-based project that utilizes OpenCV to detect redactions in declassified President's Daily Briefs from the mid-20th Century.

![](beforeafter.jpg)


# Analyzing Redactions

## Getting Started

1. Prepare two directories:
- A directory containing _only_ pdf files of PDBs titled __from_pdbs__, which is referred to as the "from" directory
- An empty directory, the "to" directory, called __to_pdbs__

As the script analyzes each PDB, it will move the files from the "from" directory to the "to" directory. This is a safety measure; if the script gets interrupted at any point, you can pick up where you left off and avoid reanalyzing any PDBs. 

2. Create an empty csv file called "__pdb_output.csv__". This will be filled with the data from the analyzed PDBs. 

## Running the Script

```bash
python3 pdb_stats.py
```
You should expect to see a table being generated on the screen. After all of the PDBs have been analyzed, you should also expect graphs to appear in a pop-up window. Note that the raw data will be stored in __pdb_output.csv__ for future use if necessary.

# Analyzing and Displaying a Single Page

Analyzing the redactions on a single page of a PDB file and display an image with the identified redactions.

## Running the Script

```bash
python3 redactions_show.py jpg_filepath
```
- __jpg_filepath__ is the filepath for a single page of a PDB. It must be a .jpg file.
- Running this script will calculate the number of redactions, the percent of text on the page that was redacted, as well as the estimated number of words that were redacted. It will also open a pop-up window containing an image that clearly identifies the locations of the redactions on the page.
- Exiting out of the pop-up window will automatically take a screenshot so you can save the analyzed image for future reference.


