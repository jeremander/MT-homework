#!/usr/bin/env python
import optparse
import sys
import copy
from collections import defaultdict
import nltk
import numpy as np
import cPickle
import itertools
import pdb
import time

ZERO_THRESH = 1e-12
RENORMALIZATION_THRESH = 1e-6
np.seterr(divide = 'ignore')

f_data = "data/hansards.f"
e_data = "data/hansards.e"
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:1000]]
bitext = [[[x.lower() for x in pair[0]], [x.lower() for x in pair[1]]] for pair in bitext]
f_vocab = sorted(list(set(nltk.flatten([pair[0] for pair in bitext]))))
e_vocab = sorted(list(set(nltk.flatten([pair[1] for pair in bitext]))))
f_lengths = map(len, [pair[0] for pair in bitext])
e_dist = nltk.MutableProbDist(nltk.UniformProbDist(e_vocab), e_vocab, True)
t_init = nltk.DictionaryConditionalProbDist(dict((f, copy.deepcopy(e_dist)) for f in f_vocab))  # uniform translation distribution
pi_init = defaultdict()
s_init = defaultdict()
for length in set(f_lengths):
    pi_init[length] = nltk.MutableProbDist(nltk.UniformProbDist(range(length)), range(length), True)
    s_init[length] = nltk.MutableProbDist(nltk.UniformProbDist(range(-length + 1, length)), range(-length + 1, length), True)
pi, s, t = copy.deepcopy(pi_init), copy.deepcopy(s_init), copy.deepcopy(t_init)

def log2_transition_prob(s, f_len, ii, i):
    """Returns log2 of transition prob from ii to i given foreign sentence length f_len. This is equal to s(i - ii) / \sum_{l = 0}^{f_len - 1} s(l - ii)."""
    denom = 0.0
    for l in xrange(f_len):
        denom += s[f_len].prob(l - ii)
    return s[f_len].logprob(i - ii) - np.log2(denom)

def ensure_normalization(dist):
    """Checks that distribution sums are sufficiently close to 1. If the residual is smaller than ZERO_THRESH, do nothing. If the residual is bigger than ZERO_THRESH, renormalize the distribution. If the residual is bigger than RENORMALIZATION_THRESH, enter debugger."""
    for key in dist.keys():
        prob_sum = sum(map(lambda x : np.power(2.0, x), dist[key]._data.tolist()))
        residual = abs(prob_sum - 1.0)
        if (residual > RENORMALIZATION_THRESH):
            pdb.set_trace()
        if (residual > ZERO_THRESH):
            log2_norm_factor = -np.log2(prob_sum)
            for samp in dist[key]._samples:
                dist[key].update(samp, dist[key].logprob(samp) + log2_norm_factor, True)

def EM_step(pi, s, t, sentence_pairs, f_vocab, e_vocab):
    """Takes translation distribution and list of pairs of sentences, and updates the translation distribution (in place)using the EM algorithm. pi is dictionary of initial distributions on states (foreign word positions), indexed by foreign sentence length. s is dictionary of distributions on transition displacements, indexed by foreign sentence length. t is conditional distribution on English words given foreign words."""
    local_f_vocab, local_e_vocab = set(), set()
    for pair in sentence_pairs:
        local_f_vocab = local_f_vocab.union(set(pair[0]))
        local_e_vocab = local_e_vocab.union(set(pair[1]))
    times = dict((key, 0.0) for key in ['setup', 'alpha', 'beta', 'gamma_xi', 'update', 'final_update'])
    start = time.time()
    f_lengths, e_lengths = [len(pair[0]) for pair in sentence_pairs], [len(pair[1]) for pair in sentence_pairs]
    f_length_counts = dict((f_len, f_lengths.count(f_len)) for f_len in set(f_lengths))
    ctr = dict((f_len, 0) for f_len in set(f_lengths))
    pi2 = defaultdict()
    s2 = dict((f_len, dict((d, 0.0) for d in xrange(-f_len + 1, f_len))) for f_len in f_lengths)
    #s3 = dict((f_len, dict(((ii, i), 0.0) for (ii, i) in itertools.product(xrange(f_len), xrange(f_len)))) for f_len in f_lengths)
    for f_len in set(f_lengths):
        pi2[f_len] = dict((i, 0.0) for i in xrange(f_len))
    t2 = dict(((f, e), 0.0) for (f, e) in itertools.product(f_vocab, e_vocab))
    t2_denom = dict((f, 0.0) for f in f_vocab)
    times['setup'] = time.time() - start
    for k in xrange(len(sentence_pairs)):
        f_sent, e_sent = sentence_pairs[k]
        f_len, e_len = f_lengths[k], e_lengths[k]
        # Forward pass
        start = time.time()
        log_alpha = np.zeros((f_len, e_len), dtype = float)  # rows are state indices, cols are output indices
        for i in xrange(f_len):
            log_alpha[i, 0] = pi[f_len].logprob(i) + t[f_sent[i]].logprob(e_sent[0])
        for j in xrange(1, e_len):
            for i in xrange(f_len):
                log_alpha[i, j] = t[f_sent[i]].logprob(e_sent[j]) + np.log2(sum([np.power(2.0, log_alpha[ii, j - 1] + log2_transition_prob(s, f_len, ii, i)) for ii in xrange(f_len)]))
        times['alpha'] += time.time() - start
        # Backward pass
        start = time.time()
        log_beta = np.zeros((f_len, e_len), dtype = float)  # rows are state indices, cols are output indices
        for j in reversed(xrange(e_len - 1)):
            for i in xrange(f_len):
                log_beta[i, j] = np.log2(sum([np.power(2.0, log_beta[ii, j + 1] + log2_transition_prob(s, f_len, i, ii) + t[f_sent[ii]].logprob(e_sent[j + 1])) for ii in xrange(f_len)]))
        times['beta'] += time.time() - start
        # Compute gammas and xis
        start = time.time()
        log_gamma = np.zeros((f_len, e_len), dtype = float)
        log_xi = np.zeros((f_len, f_len, e_len - 1), dtype = float)
        for j in xrange(e_len):
            denom = 0.0
            for i in xrange(f_len):
                term = log_alpha[i, j] + log_beta[i, j]
                log_gamma[i, j] = term
                denom += np.power(2.0, term)
                if (j < e_len - 1):
                    for ii in xrange(f_len):
                        log_xi[ii, i, j] = log_alpha[ii, j] + log_beta[i, j + 1] + log2_transition_prob(s, f_len, ii, i) + t[f_sent[i]].logprob(e_sent[j + 1])
            log_denom = np.log2(denom)
            log_gamma[:, j] -= log_denom
            if (j < e_len - 1):
                log_xi[:, :, j] -= log_denom
        times['gamma_xi'] += time.time() - start
        # debug if gamma or xi are not normalized properly
        if (not all([abs(np.power(2.0, log_gamma[:, j]).sum() - 1.0) < ZERO_THRESH for j in xrange(e_len)])):
            pdb.set_trace()
        if (not all([abs(np.power(2.0, log_xi[:,:,j]).sum() - 1.0) < ZERO_THRESH for j in xrange(e_len - 1)])):
            pdb.set_trace()
        # Update parameters
        start = time.time()
        ctr[f_len] += e_len - 1
        for i in xrange(f_len):
            pi2[f_len][i] += np.power(2.0, log_gamma[i, 0])
            for ii in xrange(f_len):
                for j in xrange(e_len - 1):
                    s2[f_len][i - ii] += np.power(2.0, log_xi[ii, i, j])
                    #s3[f_len][(ii, i)] += np.power(2.0, log_xi[ii, i, j])
            for j in xrange(e_len):
                f, e = f_sent[i], e_sent[j]
                term = np.power(2.0, log_gamma[i, j])
                t2[(f, e)] += term
                t2_denom[f] += term
        times['update'] += time.time() - start
    start = time.time()
    max_err = 0.0
    for f_len in set(f_lengths):
        for i in xrange(f_len):
            entry = pi2[f_len][i] / f_length_counts[f_len]
            err = abs(pi[f_len].prob(i) - entry)
            max_err = max(max_err, err)
            pi[f_len].update(i, np.log2(entry), True)
        s2_sum = 0.0
        for d in xrange(-f_len + 1, f_len):
            s2[f_len][d] = np.log2(s2[f_len][d]) - np.log2(ctr[f_len]) + np.log2(f_len) - np.log2(f_len - abs(d))  # for proper ratios
            s2_sum += np.power(2.0, s2[f_len][d])
        log2_s2_sum = np.log2(s2_sum)  # for normalization
        for d in xrange(-f_len + 1, f_len):
            entry = s2[f_len][d] - log2_s2_sum
            err = abs(s[f_len].prob(d) - np.power(2.0, entry))
            max_err = max(max_err, err)
            s[f_len].update(d, entry, True)
        #for (ii, i) in itertools.product(xrange(f_len), xrange(f_len)):
        #    s3[f_len][(ii, i)] = np.log2(s3[f_len][(ii, i)]) - np.log2(ctr[f_len])
        #for d in xrange(-f_len + 1, f_len):
        #    s2[f_len][d] = np.log2(s2[f_len][d]) - np.log2(ctr[f_len])
    #return (s2, s3)
    for f in f_vocab:
        if (f not in local_f_vocab):
            continue
        log2_denom = np.log2(t2_denom[f])
        for e in e_vocab:
            if (e not in local_e_vocab):
                t[f].update(e, -float('inf'), True)
            else:
                entry = np.log2(t2[(f, e)]) - log2_denom
                err = abs(t[f].prob(e) - np.power(2.0, entry))
                max_err = max(max_err, err) 
                t[f].update(e, entry, True)
    ensure_normalization(pi)
    ensure_normalization(s)
    ensure_normalization(t)
    times['final_update'] = time.time() - start
    return (log_alpha, log_beta, log_gamma, log_xi, max_err, times)


def EM(pi, s, t, sentence_pairs, err_thresh = .00001, max_iterations = 50, verbose = True, f_vocab = None, e_vocab = None):
    """Wrapper for iterating EM algorithm until convergence criterion is met. err_thresh is the maximum error (abs diff in probability entry) to allow for convergence. Otherwise if max_iterations >= 0, will stop after that many iterations. verbosity (0 = None, 1 = show iterations/errors, 2 = show steps and timings)."""
    if (f_vocab is None):
        f_vocab = sorted(list(set(nltk.flatten([pair[0] for pair in sentence_pairs]))))
    if (e_vocab is None):
        e_vocab = sorted(list(set(nltk.flatten([pair[1] for pair in sentence_pairs]))))
    ctr = 0
    err = 1.0
    if verbose:
        sys.stderr.write("%d sentence pairs\n%d foreign vocab words\n%d English vocab words\n" % (len(sentence_pairs), len(f_vocab), len(e_vocab)))
    while (err > err_thresh):
        if ((max_iterations >= 0) and (ctr >= max_iterations)):
            break
        if verbose:
            sys.stderr.write("\nIteration %d...\n" % (ctr + 1))
        (log_alpha, log_beta, log_gamma, log_xi, err, times) = EM_step(pi, s, t, sentence_pairs, f_vocab, e_vocab)
        if verbose:
            sys.stderr.write("--------------------\nTimes\nsetup:        %.5f s\nalpha:        %.5f s\nbeta:         %.5f s\ngamma & xi:   %.5f s\nupdate:       %.5f s\nfinal update: %.5f s\n--------------------\n" % (times['setup'], times['alpha'], times['beta'], times['gamma_xi'], times['update'], times['final_update']))
            sys.stderr.write("Max error: %f\n" % err)
        ctr += 1

def best_gamma_sequence(pi, s, t, sentence_pair):
    """For sentence pair (f, e), computes gammas and chooses alignment to correspond to the largest gammas for each English index."""
    f_sent, e_sent = sentence_pair
    f_len, e_len = len(sentence_pair[0]), len(sentence_pair[1])
    alignment = []
    # Forward pass
    log_alpha = np.zeros((f_len, e_len), dtype = float)  # rows are state indices, cols are output indices
    for i in xrange(f_len):
        log_alpha[i, 0] = pi[f_len].logprob(i) + t[f_sent[i]].logprob(e_sent[0])
    for j in xrange(1, e_len):
        for i in xrange(f_len):
            log_alpha[i, j] = t[f_sent[i]].logprob(e_sent[j]) + np.log2(sum([np.power(2.0, log_alpha[ii, j - 1] + log2_transition_prob(s, f_len, ii, i)) for ii in xrange(f_len)]))
    # Backward pass
    log_beta = np.zeros((f_len, e_len), dtype = float)  # rows are state indices, cols are output indices
    for j in reversed(xrange(e_len - 1)):
        for i in xrange(f_len):
            log_beta[i, j] = np.log2(sum([np.power(2.0, log_beta[ii, j + 1] + log2_transition_prob(s, f_len, ii, i) + t[f_sent[ii]].logprob(e_sent[j + 1])) for ii in xrange(f_len)]))
    # Find best gammas (don't bother normalizing)
    # for j in xrange(e_len):
    #     best_log_gamma = -float('inf')
    #     best_index = -1
    #     for i in xrange(f_len):
    #         log_gamma = log_alpha[i, j] + log_beta[i, j]
    #         if (log_gamma > best_log_gamma):
    #             best_log_gamma = log_gamma
    #             best_index = i
    #     alignment.append((best_index, j))
    log_gamma = np.zeros((f_len, e_len), dtype = float)
    for j in xrange(e_len):
        denom = 0.0
        for i in xrange(f_len):
            term = log_alpha[i, j] + log_beta[i, j]
            log_gamma[i, j] = term
            denom += np.power(2.0, term)
        log_denom = np.log2(denom)
        log_gamma[:, j] -= log_denom
        best_log_gamma = -float('inf')
        best_index = -1
        for i in xrange(f_len):
            if (log_gamma[i, j] > best_log_gamma):
                best_log_gamma = log_gamma[i, j]
                best_index = i
        alignment.append((best_index, j))
    return (log_alpha, log_beta, log_gamma, alignment)

def viterbi(pi, s, t, sentence_pair):
    """For sentence pair (f, e), computes the most likely alignment using the Viterbi algorithm."""
    f_sent, e_sent = sentence_pair
    f_len, e_len = map(len, sentence_pair)
    V = np.zeros((f_len, e_len), dtype = float)
    Ptr = np.zeros((f_len, e_len), dtype = int)
    for i in xrange(f_len):
        V[i, 0] = t[f_sent[i]].logprob(e_sent[0]) + pi[f_len].logprob(i)
        Ptr[i, 0] = i
    for j in xrange(1, e_len):
        for i in xrange(f_len):
            best_ii = -1
            best_logprob = -float('inf')
            for ii in xrange(f_len):
                logprob = t[f_sent[i]].logprob(e_sent[j]) + log2_transition_prob(s, f_len, ii, i) + V[ii, j - 1]
                if (logprob > best_logprob):
                    best_ii = ii
                    best_logprob = logprob
            V[i, j] = best_logprob
            Ptr[i, j] = best_ii
    best_states = []
    best_ii = -1
    best_logprob = -float('inf')
    for ii in xrange(f_len):
        logprob = V[ii, e_len - 1]
        if (logprob > best_logprob):
            best_ii = ii
            best_logprob = logprob
    best_states.append(best_ii)
    for j in reversed(xrange(e_len - 1)):
        best_states.append(Ptr[best_states[-1], j + 1])
    best_states.reverse()
    alignment = [(best_states[j], j) for j in xrange(e_len)]
    return alignment


def main():
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
    optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
    optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
    optparser.add_option("-i", "--iterations", dest="iterations", default=50, type="int", help="Max # of iterations (default=50)")
    optparser.add_option("-p", dest="param_filename", default=None, help="Parameter tuple pickle filename (default=None)")

    (opts, _) = optparser.parse_args()
    f_data = "%s.%s" % (opts.train, opts.french)
    e_data = "%s.%s" % (opts.train, opts.english)
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
    bitext = [[[x.lower() for x in pair[0]], [x.lower() for x in pair[1]]] for pair in bitext]
    f_lengths = map(len, [pair[0] for pair in bitext])
    load_successful = True
    try:
        (pi, s, t) = cPickle.load(open(opts.param_filename))
    except:
        f_vocab = sorted(list(set(nltk.flatten([pair[0] for pair in bitext]))))
        e_vocab = sorted(list(set(nltk.flatten([pair[1] for pair in bitext]))))
        e_dist = nltk.MutableProbDist(nltk.UniformProbDist(e_vocab), e_vocab, False)
        t = nltk.DictionaryConditionalProbDist(dict((f, copy.deepcopy(e_dist)) for f in f_vocab))
        pi = defaultdict()
        s = defaultdict()
        for length in set(f_lengths):
            pi[length] = nltk.MutableProbDist(nltk.UniformProbDist(range(length)), range(length), True)
            s[length] = nltk.MutableProbDist(nltk.UniformProbDist(range(-length + 1, length)), range(-length + 1, length), True)
    EM(pi, s, t, bitext, max_iterations = opts.iterations, verbose = True)
    cPickle.dump((pi, s, t), open("hmm_n%d_i%d.pickle" % (opts.num_sents, opts.iterations), 'wb'))
    for pair in bitext:
        alignment = best_gamma_sequence(pi, s, t, pair)[3]
        for (i, j) in alignment:
            sys.stdout.write("%i-%i " % (i, j))
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
