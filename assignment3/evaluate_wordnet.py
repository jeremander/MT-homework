#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice, product # slicing for iterators
from numpy import arange, mean, median
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

brown_ic = wordnet_ic.ic('ic-brown.dat')

ZERO_THRESH = 1e-12

def synset_similarity(synset1, synset2, metric = 'path'):
    """Returns the similarity of two synsets. Choices of similarity metric are: 'path', 'wup', 'lin'. The latter uses the Brown corpus for information content. These metrics are 0-1 normalized."""
    if (metric == 'wup'):
        return synset1.wup_similarity(synset2)
    elif (metric == 'lin'):
        return synset1.lin_similarity(synset2, brown_ic)
    else:
        return synset1.path_similarity(synset2)

def synsets_similarity(synsets1, synsets2, metric = 'path'):
    """Returns the similarity of two sets of synsets. This is the maximum similarity over all possible synset pairs."""
    if (min(len(synsets1), len(synsets2)) == 0):
        return None
    max_sim = -1.0
    for synset1 in synsets1:
        for synset2 in synsets2:
            try:
                sim = synset_similarity(synset1, synset2, metric)
                max_sim = max(max_sim, sim)
            except:
                continue
    if (max_sim < 0.0):
        return None
    return max_sim

def sentence_similarity(sent1, sent2, metric = 'path', avg = 'mean'):
    """Takes the list of words with synsets in each sentence, then for the smaller of the two lists, for each word takes the maximum word similarity with the words in the other sentence, and averages these together. Choices for averaging are 'mean' or 'median'."""
    synsetss1 = []
    for word in sent1:
        try:
            synsets = wn.synsets(word)
            if (len(synsets) > 0):
                synsetss1.append(synsets)
        except:
            continue
    synsetss2 = []
    for word in sent2:
        try:
            synsets = wn.synsets(word)
            if (len(synsets) > 0):
                synsetss2.append(synsets)
        except:
            continue
    short_list = synsetss1 if (len(synsetss1) <= len(synsetss2)) else synsetss2
    long_list = synsetss1 if (len(synsetss1) > len(synsetss2)) else synsetss2
    scores = [max([synsets_similarity(synsets1, synsets2, metric) for synsets2 in long_list]) for synsets1 in short_list]
    scores = [score for score in scores if score is not None]
    if (len(scores) == 0):
        return 0.0
    if (avg == 'mean'):
        return mean(scores)
    else:
        return median(scores)
 
def meteor_metric(hset, eset, alpha):
    """Computes P(h, e) * R(h, e) / ((1 - alpha) * R(h, e) + alpha * P(h, e)). Input h and e are sets, alpha is constant balancing precision and recall."""
    heset = hset.intersection(eset)
    P = float(len(heset)) / len(hset)
    R = float(len(heset)) / len(eset)
    return 0.0 if (max(P, R) < ZERO_THRESH) else ((P * R) / ((1 - alpha) * R + alpha * P))

def compare_with_human_evaluation(eval_file = 'eval_tune.out', answer_file = 'data/dev.answers'):
    (right, wrong) = (0.0,0.0)
    conf = [[0,0,0] for i in xrange(3)]
    for (i, (sy, sg)) in enumerate(zip(open(eval_file), open(answer_file))):
        (y, g) = (int(sy), int(sg))
        conf[g + 1][y + 1] += 1
        if g == y:
            right += 1
        else:
            wrong += 1
    acc = float(right) / (right + wrong)
    return acc
 
def sentences(input = 'data/hyp1-hyp2-ref'):
    with open(input) as f:
        for line in f:
            yield [sentence.strip().split() for sentence in line.split(' ||| ')]

def scores_from_file(input):
    with open(input) as f:
        for line in f:
            yield map(float, line.split())

def scores(sentence_gen, num_sentences, metric = 'path', avg = 'mean'):
    for h1, h2, ref in islice(sentence_gen, num_sentences):
        h1_score = sentence_similarity(h1, ref, metric, avg)
        h2_score = sentence_similarity(h2, ref, metric, avg)
        yield (h1_score, h2_score)

def result_from_score(h1_score, h2_score, thresh = .0):
    if (abs(h1_score - h2_score) < thresh):
        return 0
    elif (h1_score > h2_score):
        return 1
    else:
        return -1    

def meteor(sentence_gen, num_sentences, alpha = .76):
    for h1, h2, ref in islice(sentence_gen, num_sentences):
        h1set, h2set, rset = set(h1), set(h2), set(ref)
        h1_score = meteor_metric(h1set, rset, alpha)
        h2_score = meteor_metric(h2set, rset, alpha)
        yield (h1_score, h2_score)

def tune(weight_xrange, thresh_xrange, sent_filename = 'data/hyp1-hyp2-ref', score_filename = 'eval_scores_path_mean.out'):
    acc_results = []
    for (weight, thresh) in product(weight_xrange, thresh_xrange):
        with open('eval_tune.out', 'wb') as f:
            sc = scores_from_file(score_filename)
            met = meteor(sentences(sent_filename), None)
            for wordnet_pair in sc:
                meteor_pair = met.next()
                score1 = weight * wordnet_pair[0] + (1 - weight) * meteor_pair[0]
                score2 = weight * wordnet_pair[1] + (1 - weight) * meteor_pair[1]
                res = result_from_score(score1, score2, thresh)
                f.write(str(res) + '\n')
        acc = compare_with_human_evaluation()
        acc_results.append((weight, thresh, acc))
    return acc_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
            help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-I', '--score_input', default=None)
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    parser.add_argument('-s', '--score', action='store_true')
    parser.add_argument('-m', '--metric', default='path')
    parser.add_argument('-a', '--avg', default='mean')
    parser.add_argument('-t', '--thresh', default=.0, type=float)
    parser.add_argument('-w', '--weight', default=1.0, type=float)  # amount to weight the Wordnet score vs. the METEOR score
    opts = parser.parse_args()
    if (opts.score_input is not None):
        sc = scores_from_file(opts.score_input)
        met = meteor(sentences(opts.input), opts.num_sentences)
    else:
        sc = scores(sentences(opts.input), opts.num_sentences, opts.metric, opts.avg)
    for wordnet_pair in sc:
        meteor_pair = met.next()
        if opts.score:  # output Wordnet scores
            print(str(wordnet_pair[0]) + ' ' + str(wordnet_pair[1]))
        else:
            score1 = opts.weight * wordnet_pair[0] + (1 - opts.weight) * meteor_pair[0]
            score2 = opts.weight * wordnet_pair[1] + (1 - opts.weight) * meteor_pair[1]
            res = result_from_score(score1, score2, opts.thresh)
            print(res)
 
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
