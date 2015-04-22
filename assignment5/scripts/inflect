#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Outputs a fully inflected version of a lemmatized test set (provided on STDIN). 
If training data is provided, it will use a unigram model to select the form.

usage: cat LEMMA_FILE | python inflect
       [-t TRAINING_PREFIX] [-l LEMMA_SUFFIX] [-w WORD_SUFFIX]
"""

import argparse
import codecs
import sys
import os
from collections import defaultdict
from itertools import izip

PARSER = argparse.ArgumentParser(description="Inflect a lemmatized corpus")
PARSER.add_argument("-t", type=str, default="data/train", help="training data prefix")
PARSER.add_argument("-l", type=str, default="lemma", help="lemma file suffix")
PARSER.add_argument("-w", type=str, default="form", help="word file suffix")
args = PARSER.parse_args()

# Python sucks at UTF-8
sys.stdout = codecs.getwriter('utf-8')(sys.stdout) 
sys.stdin = codecs.getreader('utf-8')(sys.stdin) 

def inflections(lemma):
    if LEMMAS.has_key(lemma):
        return sorted(LEMMAS[lemma].keys(), lambda x,y: cmp(LEMMAS[lemma][y], LEMMAS[lemma][x]))
    return [lemma]

def best_inflection(lemma):
    return inflections(lemma)[0]

if __name__ == '__main__':

    # Build a simple unigram model on the training data
    LEMMAS = defaultdict(defaultdict)
    if args.t:
        def combine(a, b): return '%s.%s' % (a, b)
        def utf8read(file): return codecs.open(file, 'r', 'utf-8')
        # Build the LEMMAS hash, a two-level dictionary mapping lemmas to inflections to counts
        for words, lemmas in izip(utf8read(combine(args.t, args.w)), utf8read(combine(args.t, args.l))):
            for word, lemma in izip(words.rstrip().lower().split(), lemmas.rstrip().lower().split()):
                LEMMAS[lemma][word] = LEMMAS[lemma].get(word,0) + 1

    # Choose the most common inflection for each word and output them as a sentence
    for line in sys.stdin:
        print ' '.join([best_inflection(x) for x in line.rstrip().split()])
