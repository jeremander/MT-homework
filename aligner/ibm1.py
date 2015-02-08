#!/usr/bin/env python
import optparse
import sys
import copy
from collections import defaultdict
import nltk

# def main():
#     optparser = optparse.OptionParser()
#     optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
#     optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
#     optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
#     optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")

#     (opts, _) = optparser.parse_args()
#     f_data = "%s.%s" % (opts.train, opts.french)
#     e_data = "%s.%s" % (opts.train, opts.english)

#     bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

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

def EM(t, sentence_pairs, err_thresh = .000001, max_iterations = 50, verbose = True):
    f_vocab = sorted(list(set(nltk.flatten([pair[0] for pair in sentence_pairs]))))
    e_vocab = sorted(list(set(nltk.flatten([pair[1] for pair in sentence_pairs]))))
    ctr = 0
    err = 1.0
    while (err > err_thresh):
        if (ctr >= max_iterations):
            break
        if verbose:
            print("Iteration %d..." % ctr)
        err = EM_step(t, sentence_pairs, f_vocab, e_vocab)
        if verbose:
            print("Max error: %f" % err)
        ctr += 1


if __name__ == "__main__":
    main()
