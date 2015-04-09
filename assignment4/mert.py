import bleu

default_weights = {'p(e)'       : -1.0 ,
                   'p(e|f)'     : -0.5,
                   'p_lex(f|e)' : -0.5}

infile = "data/dev+test.100best"
reference = "data/dev.ref"


def rerank(weights = default_weights):
  """Function taking weights and rank sentence collections, choosing the highest-ranked choice for each sentence."""
  all_hyps = [pair.split(' ||| ') for pair in open(infile)][:40000]
  num_sents = len(all_hyps) / 100
  best_hyps = []
  for s in xrange(0, num_sents):
    hyps_for_one_sent = all_hyps[s * 100 : s * 100 + 100]
    (best_score, best) = (-1e300, '')
    for (num, hyp, feats) in hyps_for_one_sent:
      score = 0.0
      for feat in feats.split(' '):
        (k, v) = feat.split('=')
        score += weights[k] * float(v)
      if score > best_score:
        (best_score, best) = (score, hyp)
    best_hyps.append(best)
  return best_hyps

def write_hyps(hyps, filename = 'english.out'):
  with open(filename, 'w') as f:
    for hyp in hyps:
      f.write(hyp + '\n')

# def reranker():
#   feat_dict = dict()
#   all_hyps = [pair.split(' ||| ') for pair in open(infile)][:40000]
#   num_sents = len(all_hyps) / 100
#   for (num, hyp, feats) in all_hyps:
#     feat_vals = dict()
#     for feat in feats.split(' '):
#       (k, v) = feat.split('=')
#       feat_vals[k] = float(v)
#     feat_dict[hyp] = feat_vals
#   def rerank(weights = default_weights):
#     best_hyps = []
#     for s in xrange(num_sents):
#       hyps_for_one_sent = all_hyps[s * 100 : s * 100 + 100]
#       (best_score, best) = (-1e300, '')
#       for (num, hyp, feats) in hyps_for_one_sent:
#         score = 0.0
#         feat_vals = feat_dict[hyp]
#         for feat in weights.keys():
#           score += weights[feat] * feat_vals[feat]
#         if score > best_score:
#           (best_score, best) = (score, hyp)
#       best_hyps.append(best)
#     return best_hyps
#   return rerank


def fast_bleu_calculator():
  """Computes BLEU for each hypothesis paired with reference translation, and assembles these in a dictionary for fast computation."""
  all_hyps = [line.split(' ||| ')[1] for line in open(infile)]
  ref = [line.strip().split() for line in open(reference)]
  bleu_dict = dict()
  num_sents = len(all_hyps) / 100
  stats = [0 for i in xrange(10)]
  for s in xrange(min(num_sents, len(ref))):
    for i in xrange(100):
      r, h = ref[s], all_hyps[100 * s + i]
      bleu_dict[h] = list(bleu.bleu_stats(h.split(), r))
  def fast_bleu(best_hyps):
    stats = [0 for i in xrange(10)]
    for h in best_hyps:
      stats = [sum(scores) for scores in zip(stats, bleu_dict[h])]
    return bleu.bleu(stats)
  return fast_bleu

def score_best_results(best_hyps):
  """Given the list of top choices and a reference translation file, computes the aggregate BLEU score."""
  ref = [line.strip().split() for line in open(reference)]
  hyp = [sent.split() for sent in best_hyps]
  stats = [0 for i in xrange(10)]
  for (r,h) in zip(ref, hyp):
    stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(h, r))]
  return bleu.bleu(stats)

def optimize(initial_weights = default_weights, thresh = .001, verbose = True):
  """Does line search to optimize weights."""
  fast_bleu = fast_bleu_calculator()
  all_hyps = [pair.split(' ||| ') for pair in open(infile)][:40000]
  num_sents = len(all_hyps) / 100
  print("num_sents = %d" % num_sents)
  max_change = float('inf')
  weights = dict(initial_weights)
  best_weights = dict(weights)
  best_score_for_iteration = best_score = fast_bleu(rerank(best_weights))
  ctr = 1
  while (max_change > thresh):
    if verbose:
      print("Iteration #%d..." % ctr)
    best_score_for_param = best_score
    for param in weights.keys():
      print("\tparam  %s" % param)
      intersections = []
      sent_ctr = 0
      for s in xrange(num_sents):
        print("\t\tsent_ctr = %d" % sent_ctr)
        sent_ctr += 1
        hyps_for_one_sent = all_hyps[s * 100 : s * 100 + 100]
        lines = []
        hyp_ctr = 0
        for (num, hyp, feats_str) in hyps_for_one_sent:
          print("\t\t\thyp_ctr = %d" % hyp_ctr)
          hyp_ctr += 1
          feats = dict()
          for feat in feats_str.split(' '):
            feat = feat.split('=')
            (k, v) = (feat[0], float(feat[1]))
            feats[k] = v
          slope = feats[param]
          intercept = 0.0
          for other_param in weights.keys():
            if (other_param != param):
              intercept += weights[other_param] * feats[other_param]
          lines.append((slope, intercept))
        print("\t\t%d lines" % len(lines))
        lines.sort(key = lambda pair : pair[1], reverse = True)  # want highest of parallel lines
        lines.sort(key = lambda pair : pair[0])  # want steepest descent line first
        done = False
        i = 0
        while True:
          current_line = lines[i]
          (best_index, first_intersection) = (i, float('inf'))
          for j in xrange(i + 1, len(lines)):
            try:
              intersection = float(lines[j][1] - lines[i][1]) / (lines[i][0] - lines[j][0])
            except ZeroDivisionError:
              intersection = float('inf')
            if (intersection < first_intersection):
              (best_index, first_intersection) = (j, intersection)
          if (best_index > i):
            intersections.append(first_intersection)
            i = best_index
            print("\t\t\t%d" % best_index)
          else:
            break
      intersections.sort()
      print(intersections)
      test_pts = [intersections[0] - 1.0] + [(intersections[i + 1] + intersections[i]) / 2.0 for i in xrange(len(intersections) - 1)] + [intersections[-1] + 1.0]
      best_pt = weights[param]
      print("\t\t%d test points." % len(test_pts))
      for i in xrange(len(test_pts)):
        weights[param] = test_pts[i]
        score = fast_bleu(rerank(weights))
        if (score > best_score_for_param):
          best_pt, best_score_for_param = test_pts[i], score
        print("\t\t\t%d: best pt = %f, best score = %f" % (i, best_pt, best_score_for_param))
      if (best_score_for_param > best_score_for_iteration):
        weights[param] = best_pt
        best_score_for_iteration = best_score_for_param
      else:
        weights[param] = best_weights[param]
    max_change = best_score_for_iteration - best_score
    print("max change for iteration = %f" % max_change)
    if (max_change > 0.0):
      best_weights = dict(weights)
      best_score = best_score_for_iteration
    ctr += 1
  return weights

best_weights = {'p(e)': -0.48147531205132743,
 'p(e|f)': -0.2850038388607321,
 'p_lex(f|e)': -0.6070180597773039}


