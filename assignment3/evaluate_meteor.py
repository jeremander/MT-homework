#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice, product # slicing for iterators
from numpy import arange

ZERO_THRESH = 1e-12
 
def meteor_metric(hset, eset, alpha):
    """Computes P(h, e) * R(h, e) / ((1 - alpha) * R(h, e) + alpha * P(h, e)). Input h and e are sets, alpha is constant balancing precision and recall."""
    heset = hset.intersection(eset)
    P = float(len(heset)) / len(hset)
    R = float(len(heset)) / len(eset)
    return 0.0 if (max(P, R) < ZERO_THRESH) else ((P * R) / ((1 - alpha) * R + alpha * P))

def compare_with_human_evaluation(eval_file = 'eval.out', answer_file = 'data/dev.answers'):
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
 
def sentences(input):
    with open(input) as f:
        for pair in f:
            yield [sentence.strip().split() for sentence in pair.split(' ||| ')]

def results(sentence_gen, num_sentences, alpha = .76, thresh = .0):
    for h1, h2, ref in islice(sentence_gen, num_sentences):
        h1set, h2set, rset = set(h1), set(h2), set(ref)
        h1_score = meteor_metric(h1set, rset, alpha)
        h2_score = meteor_metric(h2set, rset, alpha)
        if (abs(h1_score - h2_score) < thresh):
            yield 0
        elif (h1_score > h2_score):
            yield 1
        else:
            yield -1

def tune(alpha_xrange, thresh_xrange):
    acc_results = []
    for (alpha, thresh) in product(alpha_xrange, thresh_xrange):
        with open('eval.out', 'wb') as f:
            sentence_gen = sentences('data/hyp1-hyp2-ref')
            res = results(sentence_gen, None, alpha, thresh)
            for i in res:
                f.write(str(i) + '\n')
        acc = compare_with_human_evaluation()
        acc_results.append((alpha, thresh, acc))
    return acc_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
            help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()
    res = results(sentences(opts.input), opts.num_sentences)
    for i in res:
        print(i)
 
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
