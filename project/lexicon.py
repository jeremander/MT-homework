from nltk.sem.logic import *
from nltk.featstruct import Feature, FeatStruct, FeatStructReader, FeatDict
from nltk.grammar import FeatStructNonterminal

read_expr = Expression.fromstring


##############
# MORPHOLOGY #
##############

feature_type = Feature('type')
fs_reader = FeatStructReader()
pnoun1_featstruct                = FeatStructNonterminal(fs_reader.fromstring("D[NUM=sg, PERSON=1, TRACE=False, +PropN]"))
pnoun3_featstruct                = FeatStructNonterminal(fs_reader.fromstring("D[NUM=sg, PERSON=3, TRACE=False, +PropN]"))
quant_sg_featstruct              = FeatStructNonterminal(fs_reader.fromstring("D[NUM=sg, PERSON=3, TRACE=False, -PropN]"))
quant_pl_featstruct              = FeatStructNonterminal(fs_reader.fromstring("D[NUM=pl, PERSON=3, TRACE=False, -PropN]"))
v1_sg_pres_intrans_featstruct    = FeatStructNonterminal(fs_reader.fromstring("V[TENSE=pres, NUM=sg, PERSON=1, SUBCAT=intrans]"))
v3_sg_pres_intrans_featstruct    = FeatStructNonterminal(fs_reader.fromstring("V[TENSE=pres, NUM=sg, PERSON=3, SUBCAT=intrans]"))
v_pl_pres_intrans_featstruct     = FeatStructNonterminal(fs_reader.fromstring("V[TENSE=pres, NUM=pl, SUBCAT=intrans]"))
v1_sg_pres_trans_featstruct      = FeatStructNonterminal(fs_reader.fromstring("V[TENSE=pres, NUM=sg, PERSON=1, SUBCAT=trans]"))
v3_sg_pres_trans_featstruct      = FeatStructNonterminal(fs_reader.fromstring("V[TENSE=pres, NUM=sg, PERSON=3, SUBCAT=trans]"))
v_pl_pres_trans_featstruct       = FeatStructNonterminal(fs_reader.fromstring("V[TENSE=pres, NUM=pl, SUBCAT=trans]"))
v1_sg_pres_link_featstruct       = FeatStructNonterminal(fs_reader.fromstring("V[TENSE=pres, NUM=sg, PERSON=1, SUBCAT=link]"))
v3_sg_pres_link_featstruct       = FeatStructNonterminal(fs_reader.fromstring("V[TENSE=pres, NUM=sg, PERSON=3, SUBCAT=link]"))
v_pl_pres_link_featstruct        = FeatStructNonterminal(fs_reader.fromstring("V[TENSE=pres, NUM=pl, SUBCAT=link]"))
v_past_intrans_featstruct        = FeatStructNonterminal(fs_reader.fromstring("V[TENSE=past, SUBCAT=intrans]"))
v_past_trans_featstruct          = FeatStructNonterminal(fs_reader.fromstring("V[TENSE=past, SUBCAT=trans]"))
v1_sg_past_link_featstruct       = FeatStructNonterminal(fs_reader.fromstring("V[TENSE=past, NUM=sg, PERSON=1, SUBCAT=link]"))
v3_sg_past_link_featstruct       = FeatStructNonterminal(fs_reader.fromstring("V[TENSE=past, NUM=sg, PERSON=3, SUBCAT=link]"))
v_pl_past_link_featstruct        = FeatStructNonterminal(fs_reader.fromstring("V[TENSE=past, NUM=pl, SUBCAT=link]"))
cnoun_sg_featstruct              = FeatStructNonterminal(fs_reader.fromstring("N[NUM=sg]"))
cnoun_pl_featstruct              = FeatStructNonterminal(fs_reader.fromstring("N[NUM=pl]"))
adj_featstruct                   = FeatStructNonterminal(fs_reader.fromstring("Adj"))

morphology_dict = {'JOHN'     : {pnoun3_featstruct : ['John']},
                   'MARY'     : {pnoun3_featstruct : ['Mary']},
                   'ME'        : {pnoun1_featstruct : ['I']},  # ignores case for now
                   'FARMER'   : {cnoun_sg_featstruct : ['farmer'], cnoun_pl_featstruct : ['farmers']},
                   'DONKEY'   : {cnoun_sg_featstruct : ['donkey'], cnoun_pl_featstruct : ['donkeys']},
                   'FLOWER'   : {cnoun_sg_featstruct : ['flower'], cnoun_pl_featstruct : ['flowers']},
                   'NOSE'     : {cnoun_sg_featstruct : ['nose'], cnoun_pl_featstruct : ['noses']},
                   'HAPPY'    : {adj_featstruct : ['happy']},
                   'HEALTHY'  : {adj_featstruct : ['healthy']},
                   'BUSY'     : {adj_featstruct : ['busy']},
                   'LATE'     : {adj_featstruct : ['late']},
                   'RUN'      : {v1_sg_pres_intrans_featstruct : ['run'], v3_sg_pres_intrans_featstruct : ['runs'], v_pl_pres_intrans_featstruct : ['run']},
                   'REJOICE'  : {v1_sg_pres_intrans_featstruct : ['rejoice'], v3_sg_pres_intrans_featstruct : ['rejoices'], v_pl_pres_intrans_featstruct : ['rejoice']},
                   'SING'     : {v1_sg_pres_intrans_featstruct : ['sing'], v3_sg_pres_intrans_featstruct : ['sings'], v_pl_pres_intrans_featstruct : ['sing']},
                   'SLEEP'    : {v1_sg_pres_intrans_featstruct : ['sleep'], v3_sg_pres_intrans_featstruct : ['sleeps'], v_pl_pres_intrans_featstruct : ['sleep']},
                   'RAN'      : {v_past_intrans_featstruct : ['ran']},
                   'REJOICED' : {v_past_intrans_featstruct : ['rejoiced']},
                   'SANG'     : {v_past_intrans_featstruct : ['sang']},
                   'SLEPT'    : {v_past_intrans_featstruct : ['slept']},
                   'SEE'      : {v1_sg_pres_trans_featstruct : ['see'], v3_sg_pres_trans_featstruct : ['sees'], v_pl_pres_trans_featstruct : ['see']},
                   'LIKE'     : {v1_sg_pres_trans_featstruct : ['like'], v3_sg_pres_trans_featstruct : ['likes'], v_pl_pres_trans_featstruct : ['like']},
                   'HAVE'     : {v1_sg_pres_trans_featstruct : ['have'], v3_sg_pres_trans_featstruct : ['has'], v_pl_pres_trans_featstruct : ['have']},
                   'SAW'      : {v_past_trans_featstruct : ['saw']},
                   'LIKED'    : {v_past_trans_featstruct : ['liked']},
                   'HAD'      : {v_past_trans_featstruct : ['had']},
                   'BE'       : {v1_sg_pres_link_featstruct : ['am'], v3_sg_pres_link_featstruct : ['is'], v_pl_pres_link_featstruct : ['are'], v1_sg_past_link_featstruct : ['was'], v3_sg_past_link_featstruct : ['was'], v_pl_past_link_featstruct : ['were']},
                   'EVERY'    : {quant_sg_featstruct : ['every'], quant_pl_featstruct : ['all']},
                   'SOME'     : {quant_sg_featstruct : ['a']},
                   'THE'      : {quant_sg_featstruct : ['the']},  # can't handle plurals yet
                   'THIS'     : {quant_sg_featstruct : ['this']},
                   'NO'       : {quant_sg_featstruct : ['no'], quant_pl_featstruct : ['no']}}
default_featstructs = {'TP'         : FeatStructNonterminal(fs_reader.fromstring("TP[TENSE=?t, NUM=?n, PERSON=?p]")),
                       'TBar'       : FeatStructNonterminal(fs_reader.fromstring("TBar[TENSE=?t, NUM=?n, PERSON=?p]")),
                       'T'          : FeatStructNonterminal(fs_reader.fromstring("T[TENSE=?t, NUM=?n, PERSON=?p, AUX=?a]")),
                       'PA'         : FeatStructNonterminal(fs_reader.fromstring("PA[TENSE=?t, NUM=?n, PERSON=?p]")),
                       'VP'         : FeatStructNonterminal(fs_reader.fromstring("VP[TENSE=?t, NUM=?n, PERSON=?p]")),
                       'VBar'       : FeatStructNonterminal(fs_reader.fromstring("VBar[TENSE=?t, NUM=?n, PERSON=?p]")),
                       'V_intrans'  : FeatStructNonterminal(fs_reader.fromstring("V[TENSE=?t, NUM=?n, PERSON=?p, SUBCAT=intrans]")),
                       'V_trans'    : FeatStructNonterminal(fs_reader.fromstring("V[TENSE=?t, NUM=?n, PERSON=?p, SUBCAT=trans]")),
                       'V_link'     : FeatStructNonterminal(fs_reader.fromstring("V[TENSE=?t, NUM=?n, PERSON=?p, SUBCAT=link]")),
                       'DP_PropN'   : FeatStructNonterminal(fs_reader.fromstring("DP[NUM=sg, PERSON=?p, -TRACE, +PropN]")),
                       'DP_quant'   : FeatStructNonterminal(fs_reader.fromstring("DP[NUM=?n, PERSON=?p, -TRACE, -PropN]")),
                       'DP_trace'   : FeatStructNonterminal(fs_reader.fromstring("DP[NUM=?n, PERSON=?p, +TRACE, -PropN]")),
                       'DBar_PropN' : FeatStructNonterminal(fs_reader.fromstring("DBar[NUM=sg, PERSON=?p, -TRACE, +PropN]")),
                       'DBar_quant' : FeatStructNonterminal(fs_reader.fromstring("DBar[NUM=?n, PERSON=?p, -TRACE, -PropN]")),
                       'D_PropN'    : FeatStructNonterminal(fs_reader.fromstring("D[NUM=sg, PERSON=?p, -TRACE, +PropN]")),
                       'D_quant'    : FeatStructNonterminal(fs_reader.fromstring("D[NUM=?n, PERSON=?p, -TRACE, -PropN]")),
                       'NP'         : FeatStructNonterminal(fs_reader.fromstring("NP[NUM=?n]")),
                       'NBar'       : FeatStructNonterminal(fs_reader.fromstring("NBar[NUM=?n]")),
                       'N'          : FeatStructNonterminal(fs_reader.fromstring("N[NUM=?n]")),
                       'AdjP'       : FeatStructNonterminal(fs_reader.fromstring("AdjP[]")),
                       'AdjBar'     : FeatStructNonterminal(fs_reader.fromstring("AdjBar[]")),
                       'Adj'        : FeatStructNonterminal(fs_reader.fromstring("Adj[]"))}

###################
# ENGLISH LEXICON #
###################

proper_nouns = ['JOHN', 'MARY', 'ME']
common_nouns = ['FARMER', 'DONKEY', 'FLOWER', 'NOSE']
adjectives = ['HAPPY', 'HEALTHY', 'BUSY', 'LATE']
intransitive_verbs = ['RUN', 'REJOICE', 'SING', 'SLEEP', 'RAN', 'REJOICED', 'SANG', 'SLEPT']
transitive_verbs = ['SEE', 'LIKE', 'HAVE', 'SAW', 'LIKED', 'HAD']
quantifiers = ['EVERY', 'SOME', 'THE', 'THIS', 'NO']

english_signature_types = {'DP' : '<<e, t>, t>', 'Det' : '<<e, t>, <<e, t>, t>>', 'Pred' : '<e, t>', 'TransVerb' : '<e, <e, t>>', 'PredLink' : '<<e, t>, <e, t>>', 'DPLink' : '<e, <e, t>>'}
english_signature = {'ME' : 'e', 'JOHN' : 'e', 'MARY' : 'e', 'UNIQUE' : '<<e, t>, t>', 'NEAR' : '<e, t>', 'MOST' : '<<e, t>, <<e, t>, t>>', 'RUN' : '<e, t>', 'REJOICE' : '<e, t>', 'SING' : '<e, t>', 'SLEEP' : '<e, t>', 'SEE' : '<e, <e, t>>', 'LIKE' : '<e, <e, t>>', 'HAVE' : '<e, <e, t>>', 'RAN' : '<e, t>', 'REJOICED' : '<e, t>', 'SANG' : '<e, t>', 'SLEPT' : '<e, t>', 'SAW' : '<e, <e, t>>', 'LIKED' : '<e, <e, t>>', 'HAD' : '<e, <e, t>>', 'FARMER' : '<e, t>', 'DONKEY' : '<e, t>', 'FLOWER' : '<e, t>', 'NOSE' : '<e, t>', 'HAPPY' : '<e, t>', 'HEALTHY' : '<e, t>', 'BUSY' : '<e, t>', 'LATE' : '<e, t>', 'x' : 'e', 'y' : 'e', 'P' : '<e, t>', 'Q' : '<e, t>'}
english_lexicon = {
    'I'        : {'DP'  : [r'\P . P(ME)']},
    'John'     : {'DP'  : [r'\P . P(JOHN)']},
    'Mary'     : {'DP'  : [r'\P . P(MARY)']},
    'the'      : {'Det' : [r'\P . \Q . (exists x . (UNIQUE(P) & P(x) & Q(x)))']},
    'a'        : {'Det' : [r'\P . \Q . (exists x . (P(x) & Q(x)))']},
    'this'     : {'Det' : [r'\P . \Q . (exists x . (UNIQUE(P) & NEAR(x) & P(x) & Q(x)))']},
    'every'    : {'Det' : [r'\P . \Q . (all x . (P(x) -> Q(x)))']},
    'no'       : {'Det' : [r'\P . \Q . -(exists x . (P(x) & Q(x)))']},
    'all'      : {'Det' : [r'\P . \Q . (all x . (P(x) -> Q(x)))']},
    'most'     : {'Det' : [r'\P . \Q . MOST(P)(Q)']},
    'run'      : {'Pred' : [r'\x . RUN(x)']},
    'rejoice'  : {'Pred' : [r'\x . REJOICE(x)']},
    'sing'     : {'Pred' : [r'\x . SING(x)']},
    'sleep'    : {'Pred' : [r'\x . SLEEP(x)']},
    'runs'     : {'Pred' : [r'\x . RUN(x)']},
    'rejoices' : {'Pred' : [r'\x . REJOICE(x)']},
    'sings'    : {'Pred' : [r'\x . SING(x)']},
    'sleeps'   : {'Pred' : [r'\x . SLEEP(x)']},
    'see'      : {'TransVerb' : [r'\x . (\y . SEE(x)(y))']},
    'like'     : {'TransVerb' : [r'\x . (\y . LIKE(x)(y))']},
    'have'     : {'TransVerb' : [r'\x . (\y . HAVE(x)(y))']},
    'sees'     : {'TransVerb' : [r'\x . (\y . SEE(x)(y))']},
    'likes'    : {'TransVerb' : [r'\x . (\y . LIKE(x)(y))']},
    'has'      : {'TransVerb' : [r'\x . (\y . HAVE(x)(y))']},
    'am'       : {'PredLink' : [r'\P . P'], 'DPLink' : [r'\x . (\y . (y == x))']},
    'is'       : {'PredLink' : [r'\P . P'], 'DPLink' : [r'\x . (\y . (y == x))']},
    'are'      : {'PredLink' : [r'\P . P'], 'DPLink' : [r'\x . (\y . (y == x))']},
    'ran'      : {'Pred' : [r'\x . RAN(x)']},
    'rejoiced' : {'Pred' : [r'\x . REJOICED(x)']},
    'sang'     : {'Pred' : [r'\x . SANG(x)']},
    'slept'    : {'Pred' : [r'\x . SLEPT(x)']},
    'saw'      : {'TransVerb' : [r'\x . (\y . SAW(x)(y))']},
    'liked'    : {'TransVerb' : [r'\x . (\y . LIKED(x)(y))']},
    'had'      : {'TransVerb' : [r'\x . (\y . HAD(x)(y))']},
    'was'      : {'PredLink' : [r'\P . P'], 'DPLink' : [r'\x . (\y . (y == x))']},
    'were'     : {'PredLink' : [r'\P . P'], 'DPLink' : [r'\x . (\y . (y == x))']},
    'farmer'   : {'Pred' : [r'\x . FARMER(x)']},
    'donkey'   : {'Pred' : [r'\x . DONKEY(x)']},
    'flower'   : {'Pred' : [r'\x . FLOWER(x)']},
    'nose'     : {'Pred' : [r'\x . NOSE(x)']},
    'farmers'  : {'Pred' : [r'\x . FARMER(x)']},
    'donkeys'  : {'Pred' : [r'\x . DONKEY(x)']},
    'flowers'  : {'Pred' : [r'\x . FLOWER(x)']},
    'noses'    : {'Pred' : [r'\x . NOSE(x)']},
    'happy'    : {'Pred' : [r'\x . HAPPY(x)']},
    'healthy'  : {'Pred' : [r'\x . HEALTHY(x)']},
    'busy'     : {'Pred' : [r'\x . BUSY(x)']},
    'late'     : {'Pred' : [r'\x . LATE(x)']}}

for word in english_lexicon:
    for cat in english_lexicon[word]:
        # convert from string to logic expression
        english_lexicon[word][cat] = [read_expr(expr, type_check = True, signature = english_signature) for expr in english_lexicon[word][cat]]


####################
# JAPANESE LEXICON #
####################

japanese_signature_types = {'DP' : '<<e, t>, t>', 'Det' : '<<e, t>, <<e, t>, t>>', 'Pred' : '<e, t>', 'TransVerb' : '<e, <e, t>>', 'PredLink' : '<<e, t>, <e, t>>', 'DPLink' : '<e, <e, t>>', 'DPParticle' : '<<<e, t>, t>, <<e, t>, t>>', 'AdjParticle' : '<<e, t>, <e, t>>'}
japanese_signature = {'ME' : 'e', 'JOHN' : 'e', 'MARY' : 'e', 'UNIQUE' : '<<e, t>, t>', 'NEAR' : '<e, t>', 'MOST' : '<<e, t>, <<e, t>, t>>', 'RUN' : '<e, t>', 'REJOICE' : '<e, t>', 'SING' : '<e, t>', 'SLEEP' : '<e, t>', 'SEE' : '<e, <e, t>>', 'LIKE' : '<e, <e, t>>', 'HAVE' : '<e, <e, t>>', 'RAN' : '<e, t>', 'REJOICED' : '<e, t>', 'SANG' : '<e, t>', 'SLEPT' : '<e, t>', 'SAW' : '<e, <e, t>>', 'LIKED' : '<e, <e, t>>', 'HAD' : '<e, <e, t>>', 'FARMER' : '<e, t>', 'DONKEY' : '<e, t>', 'FLOWER' : '<e, t>', 'NOSE' : '<e, t>', 'HAPPY' : '<e, t>', 'HEALTHY' : '<e, t>', 'BUSY' : '<e, t>', 'LATE' : '<e, t>', 'x' : 'e', 'y' : 'e', 'P' : '<e, t>', 'Q' : '<e, t>', 'D' : '<<e, t>, t>'}
japanese_lexicon = {
    'watashi'         : {'DP'  : [r'\P . P(ME)']},
    'John'            : {'DP'  : [r'\P . P(JOHN)']},
    'Mary'            : {'DP'  : [r'\P . P(MARY)']},
    '*det*'           : {'Det' : [r'\P . \Q . (exists x . (P(x) & Q(x)))', r'\P . \Q . (exists x . (UNIQUE(P) & P(x) & Q(x)))']},
    'kono'            : {'Det' : [r'\P . \Q . (exists x . (UNIQUE(P) & NEAR(x) & P(x) & Q(x)))']},
    'subete-no'       : {'Det' : [r'\P . \Q . (all x . (P(x) -> Q(x)))']},
    'hashirimasu'     : {'Pred' : [r'\x . RUN(x)']},
    'yorokobimasu'    : {'Pred' : [r'\x . REJOICE(x)']},
    'utaimasu'        : {'Pred' : [r'\x . SING(x)']},
    'nemasu'          : {'Pred' : [r'\x . SLEEP(x)']},
    'okureteimasu'       : {'Pred' : [r'\x . LATE(x)']},
    'miteimasu'       : {'TransVerb' : [r'\x . (\y . SEE(x)(y))']},
    'motteimasu'      : {'TransVerb' : [r'\x . (\y . HAVE(x)(y))']},
    'desu'            : {'PredLink' : [r'\P . P'], 'DPLink' : [r'\x . (\y . (y == x))']},
    'hashirimashita'  : {'Pred' : [r'\x . RAN(x)']},
    'yorokobimashita' : {'Pred' : [r'\x . REJOICED(x)']},
    'utaimashita'     : {'Pred' : [r'\x . SANG(x)']},
    'nemashita'       : {'Pred' : [r'\x . SLEPT(x)']},
    'okuremashita'    : {'Pred' : [r'\x . LATE(x)']},
    'mimashita'       : {'TransVerb' : [r'\x . (\y . SAW(x)(y))']},
    'motteimashita'   : {'TransVerb' : [r'\x . (\y . HAD(x)(y))']},
    'deshita'         : {'PredLink' : [r'\P . P'], 'DPLink' : [r'\x . (\y . (y == x))']},
    'nouka'           : {'Pred' : [r'\x . FARMER(x)']},
    'roba'            : {'Pred' : [r'\x . DONKEY(x)']},
    'hana'            : {'Pred' : [r'\x . FLOWER(x)', r'\x . NOSE(x)']},
    'ureshii'         : {'Pred' : [r'\x . HAPPY(x)']},
    'ureshikatta'     : {'Pred' : [r'\x . HAPPY(x)']},
    'suki'            : {'Pred' : [r'\x . (LIKE(x)(y))']},  # note the free variable!
    'genki'           : {'Pred' : [r'\x . HEALTHY(x)']},
    'nigiyaka'        : {'Pred' : [r'\x . BUSY(x)']},
    'wa'              : {'DPParticle' : [r'\D . D']},
    'ga'              : {'DPParticle' : [r'\D . D']},
    'wo'              : {'DPParticle' : [r'\D . D']},
    'na'              : {'AdjParticle' : [r'\P . P']}}

for word in japanese_lexicon:
    for cat in japanese_lexicon[word]:
        # convert from string to logic expression
        japanese_lexicon[word][cat] = [read_expr(expr, type_check = True, signature = japanese_signature) for expr in japanese_lexicon[word][cat]]