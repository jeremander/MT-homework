#!/usr/bin/env python

"""
Extracts the tags, lemmas, and forms from the PDT. 

usage: pull.py /path/to/pdt/m.gz/file
"""


import xml.etree.ElementTree as ET
import argparse
import codecs
import gzip
import sys

PARSER = argparse.ArgumentParser(description="Extract data from the PDT v2.0")
PARSER.add_argument("files", type=str, nargs='+', help="(compressed) file to read")
PARSER.add_argument("-tag", type=str, default='form', help="tag name to extract (e.g., lemma, form)")
PARSER.add_argument("-tokens", default=False, action='store_true', help="output tokens one per line")
args = PARSER.parse_args()

sys.stdout = codecs.getwriter('utf-8')(sys.stdout) 

for file in args.files:
    tree = ET.parse(gzip.open(file))
    root = tree.getroot()
    for sentence in root.findall('{http://ufal.mff.cuni.cz/pdt/pml/}s'):
        for word in sentence.findall('{http://ufal.mff.cuni.cz/pdt/pml/}m'):
            tag = word.findall('{http://ufal.mff.cuni.cz/pdt/pml/}%s' % (args.tag))[0]
            print tag.text,
            if args.tokens:
                print
        print
