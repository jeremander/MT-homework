#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Outputs a fully inflected version of a lemmatized test set.
If training data is provided, it will use a unigram model to select the form.

usage: cat LEMMA_FILE | python inflect
       [-t TRAINING_PREFIX] [-l LEMMA_SUFFIX] [-w WORD_SUFFIX]
"""

import argparse
import codecs
import sys
import os
from collections import defaultdict
from itertools import izip, product
import numpy as np

DEFAULT_WEIGHTS = [0.0, 0.383, 0.307, 0.307]
alpha = 0.5


PARSER = argparse.ArgumentParser(description="Inflect a lemmatized corpus")
PARSER.add_argument("-t", type=str, default="data/train", help="training data prefix")
PARSER.add_argument("-s", type=str, default="data/dtest", help="test data prefix")
PARSER.add_argument("-l", type=str, default="lemma", help="lemma file suffix")
PARSER.add_argument("-w", type=str, default="form", help="word file suffix")
PARSER.add_argument("-g", type=str, default="tag", help="tag file suffix")
PARSER.add_argument("-gold", type=str, default='data/dtest.form', help="gold file for word forms")
args = PARSER.parse_args()

def normalize(v):
    total = sum(v)
    return [float(elt) / total for elt in v]

def unigram_inflections(lemma):
    """Given a lemma, returns dictionary of MLE probabilities for each word form."""
    if LEMMAS.has_key(lemma):
        return LEMMAS[lemma]
    return {lemma : 1}

def bigram_lemma_inflections(lemma, prev_lemma):
    if LEMMAS_LEMMAS.has_key((lemma, prev_lemma)):
        return LEMMAS_LEMMAS[(lemma, prev_lemma)]
    return {}

def bigram_word_inflections(lemma, prev_word):
    if WORDS_LEMMAS.has_key((lemma, prev_word)):
        return WORDS_LEMMAS[(lemma, prev_word)]
    return {}

def lemma_pos_inflections(lemma, pos):
    if TAGS_LEMMAS.has_key((lemma, pos)):
        return TAGS_LEMMAS[(lemma, pos)]
    return {}

def best_inflection(lemma, prev_lemma, prev_word, pos, weights = DEFAULT_WEIGHTS):
    inflection_counts = [unigram_inflections(lemma), bigram_lemma_inflections(lemma, prev_lemma), bigram_word_inflections(lemma, prev_word), lemma_pos_inflections(lemma, pos)]
    total_counts = [sum(inflection_counts[i].values()) for i in xrange(len(inflection_counts))]
    candidates = []
    for dct in inflection_counts:
        candidates.extend(dct.keys())
    candidates = list(set(candidates))
    smoothed_inflection_probs = [dict((candidate, float(inflection_counts[i][candidate] + alpha) / (total_counts[i] + alpha * len(candidates)) if (candidate in inflection_counts[i]) else float(alpha) / (total_counts[i] + alpha * len(candidates))) for candidate in candidates) for i in xrange(len(inflection_counts))]
    best_score = -float('inf')
    best_candidate = None
    for candidate in candidates:
        model_logprobs = [np.log(smoothed_inflection_probs[i][candidate]) for i in xrange(len(inflection_counts))]
        score = sum([weights[i] * model_logprobs[i] for i in xrange(len(weights))])
        if (score > best_score):
            best_score = score
            best_candidate = candidate
    return best_candidate

def combine(a, b): return '%s.%s' % (a, b)

def utf8read(file): return codecs.open(file, 'r', 'utf-8')

def equal(pair): return pair[0].lower() == pair[1].lower()


if __name__ == '__main__':

    # Python sucks at UTF-8
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout) 

    training_lemma_file = utf8read(combine(args.t, args.l))
    training_pos_file = utf8read(combine(args.t, args.g))
    training_word_file = utf8read(combine(args.t, args.w))
    test_lemma_file = utf8read(combine(args.s, args.l))
    test_pos_file = utf8read(combine(args.s, args.g))
    gold_file = codecs.open(args.gold, 'r', 'utf-8')

    LEMMAS = defaultdict(defaultdict)           # P(word_i | lemma_i)
    LEMMAS_LEMMAS = defaultdict(defaultdict)    # P(word_i | lemma_i, lemma_{i-1})
    WORDS_LEMMAS = defaultdict(defaultdict)     # P(word_i | lemma_i, word_{i-1})
    TAGS_LEMMAS = defaultdict(defaultdict)      # P(word_i | lemma_i, tag_i)

    # Build the LEMMAS hash, a two-level dictionary mapping lemmas to inflections to counts
    for lemmas, tags, words in izip(training_lemma_file, training_pos_file, training_word_file):
        lemmas = lemmas.rstrip().lower().split()
        tags = tags.rstrip().lower().split()
        words = words.rstrip().lower().split()
        for i in xrange(len(words)):
            LEMMAS[lemmas[i]][words[i]] = LEMMAS[lemmas[i]].get(words[i], 0) + 1
            TAGS_LEMMAS[(lemmas[i], tags[i])][words[i]] = TAGS_LEMMAS[(lemmas[i], tags[i])].get(words[i], 0) + 1
            if (i == 0):
                LEMMAS_LEMMAS[(lemmas[i], None)][words[i]] = LEMMAS_LEMMAS[(lemmas[i], None)].get(words[i], 0) + 1
                WORDS_LEMMAS[(lemmas[i], None)][words[i]] = WORDS_LEMMAS[(lemmas[i], None)].get(words[i], 0) + 1
            else:
                LEMMAS_LEMMAS[(lemmas[i], lemmas[i - 1])][words[i]] = LEMMAS_LEMMAS[(lemmas[i], lemmas[i - 1])].get(words[i], 0) + 1
                WORDS_LEMMAS[(lemmas[i], words[i - 1])][words[i]] = WORDS_LEMMAS[(lemmas[i], words[i - 1])].get(words[i], 0) + 1

    # Choose the best-scoring inflection for each word and output them as a sentence
    #weights_list = product(*[np.arange(0, 1.2, .2) for i in xrange(4)])
    weights_list = [DEFAULT_WEIGHTS]
    for weights in weights_list:
        test_lemma_file.seek(0)
        test_pos_file.seek(0)
#        gold_file.seek(0)
        total = 0
        right = 0
#        for lemmas, tags, gold in izip(test_lemma_file, test_pos_file, gold_file):
        for lemmas, tags in izip(test_lemma_file, test_pos_file):
            words = []
            lemmas = lemmas.rstrip().lower().split()
            tags = tags.rstrip().lower().split()
#            gold = gold.rstrip().lower().split()
            s = ''
            for i in xrange(len(lemmas)):
                prev_lemma = lemmas[i - 1] if (i > 0) else None
                prev_word = words[-1] if (i > 0) else None
                inflection = best_inflection(lemmas[i], prev_lemma, prev_word, tags[i], weights)
                s += (inflection + ' ')
                words.append(inflection)
            # compared = map(equal, izip(words, gold))
            # right += sum(compared)
            # total += len(compared)
            # grade = 1.0 * right / total
            print(s)
        # sys.stderr.write("weights = %s\n%d / %d = %.4f\n\n" % (str(weights), right, total, grade))


