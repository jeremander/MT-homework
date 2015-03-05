#!/usr/bin/env python
import optparse
import sys
import copy
from collections import defaultdict
import nltk
import numpy as np
import cPickle

ZERO_THRESH = 1e-12

f_data = "data/hansards.f"
e_data = "data/hansards.e"
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:100]]
f_vocab = sorted(list(set(nltk.flatten([pair[0] for pair in bitext]))))
e_vocab = sorted(list(set(nltk.flatten([pair[1] for pair in bitext]))))
e_dist = nltk.MutableProbDist(nltk.UniformProbDist(e_vocab), e_vocab, False)
t_init = nltk.DictionaryConditionalProbDist(dict((f, copy.deepcopy(e_dist)) for f in f_vocab))  # uniform translation distribution

def EM_step(t, sentence_pairs, f_vocab, e_vocab):
    """Takes translation distribution and list of pairs of sentences, and updates the translation distribution (in place)using the EM algorithm."""
    count = nltk.ConditionalFreqDist()
    total = nltk.FreqDist()
    for (fs, es) in sentence_pairs:
        s_total = nltk.FreqDist()
        for e in es:
            s_total[e] = 0.0
            for f in fs:
                s_total[e] += t[f].prob(e)
        for e in es:
            for f in fs:
                inc = t[f].prob(e) / s_total[e]
                count[f][e] += inc
                total[f] += inc
    max_err = 0.0  # maximum change in an entry in the translation probabilities
    for f in f_vocab:
        for e in e_vocab:
            entry = count[f][e] / total[f]
            err = abs(entry - t[f].prob(e))
            if (err > max_err):
                max_err = err
            t[f].update(e, entry, False)
    return max_err

def EM(t, sentence_pairs, err_thresh = .00001, max_iterations = 50, verbose = True):
    """Wrapper for iterating EM algorithm until convergence criterion is met. err_thresh is the maximum error (abs diff in probability entry) to allow for convergence. Otherwise if max_iterations >= 0, will stop after that many iterations."""
    f_vocab = sorted(list(set(nltk.flatten([pair[0] for pair in sentence_pairs]))))
    e_vocab = sorted(list(set(nltk.flatten([pair[1] for pair in sentence_pairs]))))
    ctr = 0
    err = 1.0
    while (err > err_thresh):
        if ((max_iterations >= 0) and (ctr >= max_iterations)):
            break
        if verbose:
            sys.stderr.write("Iteration %d...\n" % (ctr + 1))
        err = EM_step(t, sentence_pairs, f_vocab, e_vocab)
        if verbose:
            sys.stderr.write("Max error: %f\n" % err)
        ctr += 1

def most_probable_alignment(t, sentence_pair):
    """For sentence pair (f, e), for each e_j, computes the numerators of the probabilities p(a = (j, i) | e, f), then chooses the (j, i) pair with the highest probability. If there is a tie, chooses among the (j, i) such that i * (len(e) / len(f)) is closest to j."""
    alignment = []
    for j in xrange(len(sentence_pair[1])):
        length_ratio = float(len(sentence_pair[1])) / float(len(sentence_pair[0]))
        best_prob = -1.0
        best_index = -1
        for i in xrange(len(sentence_pair[0])):
            prob = t[sentence_pair[0][i]].prob(sentence_pair[1][j])
            if ((prob > best_prob + ZERO_THRESH) or ((abs(prob - best_prob) < ZERO_THRESH) and (abs(j - i * length_ratio) < abs(j - best_index * length_ratio)))):
                best_prob = prob
                best_index = i
        alignment.append((best_index, j))
    return alignment

def main():
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
    optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
    optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
    optparser.add_option("-i", "--iterations", dest="iterations", default=50, type="int", help="Max # of iterations (default=50)")
    optparser.add_option("-t", dest="t_filename", default=None, help="Translation probability pickle filename (default=None)")

    (opts, _) = optparser.parse_args()
    f_data = "%s.%s" % (opts.train, opts.french)
    e_data = "%s.%s" % (opts.train, opts.english)
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
    load_successful = True
    try:
        t = cPickle.load(open(opts.t_filename))
    except:
        f_vocab = sorted(list(set(nltk.flatten([pair[0] for pair in bitext]))))
        e_vocab = sorted(list(set(nltk.flatten([pair[1] for pair in bitext]))))
        e_dist = nltk.MutableProbDist(nltk.UniformProbDist(e_vocab), e_vocab, False)
        t = nltk.DictionaryConditionalProbDist(dict((f, copy.deepcopy(e_dist)) for f in f_vocab))
    EM(t, bitext, max_iterations = opts.iterations, verbose = True)
    cPickle.dump(t, open("ibm1_t_n%d_i%d.pickle" % (opts.num_sents, opts.iterations), 'wb'))
    for pair in bitext:
        alignment = most_probable_alignment(t, pair)
        for (i, j) in alignment:
            sys.stdout.write("%i-%i " % (i, j))
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
