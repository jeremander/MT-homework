#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Produces the counts of forms for each lemma over a parallel dataset.

usage: paste train.form train.lemma | python count.py > lemmas.txt
"""

import argparse
import codecs
import sys
from itertools import izip

PARSER = argparse.ArgumentParser(description="Count lemmas and word forms")
args = PARSER.parse_args()

sys.stdout = codecs.getwriter('utf-8')(sys.stdout) 
sys.stdin = codecs.getreader('utf-8')(sys.stdin) 

hash = {}
for line in sys.stdin:
    wordline, lemmaline = line.rstrip().split('\t')
    words = wordline.split()
    lemmas = lemmaline.split()

    for word, lemma in izip(words, lemmas):
        if not hash.has_key(lemma):
            hash[lemma] = {}
        hash[lemma][word] = hash[lemma].get(word,0) + 1

for lemma in hash.keys():
    print lemma, len(hash[lemma].keys()),
    for word in hash[lemma].keys():
        print word, hash[lemma][word],
    print
