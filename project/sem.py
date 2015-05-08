from itertools import *
from tree import *
from nltk.sem.logic import *
from lexicon import *

def flatten(llst):
    """Flattens a list one level."""
    res = []
    for lst in llst:
        res += lst
    return res

class Trace(object):
    def __init__(self, index, bound = True):
        """Takes an Index object."""
        self.index = index
        self.bound = bound
    def __str__(self):
        return ("%s%s" % ('t' if self.bound else 'x', str(self.index)))
    def __repr__(self):
        return str(self)
    def __eq__(self, other):
        return (self.index == other.index)
    def __ne__(self, other):
        return (not (self == other))
    def __cmp__(self, other):
        return cmp(self.index, other.index)
    def __hash__(self):
        return hash(self.index)


class SynTree(Tree):
    """Class for syntax trees. Each node is labeled with a unique ID. Leaf nodes contain strings that are interpretable words."""
    lexicon_dict = {'english' : english_lexicon, 'japanese' : japanese_lexicon}
    def __init__(self, label, children = [], QR_level = 0, language = 'english'):
        super(SynTree, self).__init__(label, children)
        self.QR_level = QR_level # number of times QR has been applied
        self.language = language
    def category(self):
        """Returns the grammatical category of the root node."""
        if self.is_terminal():
            return "Word"
        return self.label[feature_type]
    def draw(self, with_ID = False, filename = None):
        if (not hasattr(self, 'nx_tree')):
            self.make_nx_tree()
        labels = dict((node, str(node[0][feature_type])) if isinstance(node[0], FeatStructNonterminal) else (node, str(node[0])) for node in self.nx_tree.nodes())
        if with_ID:
            for node in labels.keys():
                labels[node] += ("\n%s" % node[1])
        for key in labels.keys():
            labels[key] = labels[key].replace("Bar", "'")
        super(SynTree, self).draw(labels, filename)
    def set_QR_level(self, n):
        """Sets QR level of all the nodes to n."""
        def set_QR_level_of_node(node, args_dict = {}):
            node.QR_level = n
        self.postorder_traverse(set_QR_level_of_node)
    def QR(self):
        """Quantifier raising (returns generator over permutations of the quantifiers)."""
        assert(self.category() == 'TP')
        if (not hasattr(self, 'nx_tree')):
            self.make_nx_tree()
        tree = deepcopy(self)
        if (tree.language == 'japanese'):
            tree.postorder_traverse(adjust_tree, {})
        DPs = tree.postorder_traverse(raise_DPs, {'DPs' : []})['DPs']
        if (self.language == 'japanese'):
            if (len([DP for DP in DPs if DP.label['PARTICLE'] == 'wa']) > 1):  # only one topic permitted
                return iter(())
            else:  # topic DP must get widest scope
                i = [(DP.label['PARTICLE'] == 'wa') for DP in DPs].index(True)
                perms = (perm for perm in permutations(xrange(len(DPs))) if (perm[0] == i))
        else:
            perms = permutations(xrange(len(DPs)))
        TP_label, PA_label = deepcopy(tree.label), deepcopy(tree.label)
        PA_label[feature_type] = 'PA'
        def tree_gen(tree):
            for perm in perms:
                pa_tree = copy(tree)
                for i in reversed(perm):  # first DP in the list gets widest scope
                    PA_subtree = SynTree(PA_label, [SynTree(DPs[i].ID, [], tree.QR_level, tree.language), copy(pa_tree)], tree.QR_level, tree.language)
                    pa_tree = SynTree(TP_label, [DPs[i], PA_subtree], tree.QR_level, tree.language)
                pa_tree = deepcopy(pa_tree)
                pa_tree.label_nodes()
                yield pa_tree
        return tree_gen(tree)
    def quantifier_lower(self):
        """First replaces any traces with the quantifying DPs that bind them. Then removes the DPs in QR position, leaving a surface structure tree."""
        tree = deepcopy(self)
        all_DPs = [subtree for subtree in tree.subtree_dict.values() if (isinstance(subtree.label, FeatStructNonterminal) and (subtree.label[feature_type] == 'DP'))]
        trace_DPs = [DP for DP in all_DPs if DP.label['TRACE']]
        non_trace_DPs = diff(all_DPs, trace_DPs)
        for DP in trace_DPs:
            for other_DP in non_trace_DPs:
                if (DP.children[0].label.index == other_DP.ID):
                    DP.label = deepcopy(other_DP.label)
                    DP.children = deepcopy(other_DP.children)
                    break
        tree = remove_QRed_DPs(tree)
        tree.label_nodes()
        tree.make_nx_tree()
        return tree
    def interpret(self):
        """Returns generator of SynTrees marked at each node with a denotation."""
        # NOTE (for later): May be a good idea to re-link the indices & traces so that the above routine can use identity instead of equality as coreference criterion
        QR_gen = self.QR() if (self.QR_level == 0) else [self]
        def tree_gen():
            for tree in QR_gen:
                interpreted_tree_gen = tree.postorder_traverse(interpret_tree, args_dict = {'lexicon' : SynTree.lexicon_dict[self.language], 'generators' : {}})
                for interpreted_tree in interpreted_tree_gen:
                    yield interpreted_tree
        return tree_gen()
    def resolve_features_bottom_up(self):
        def resolve_features_at_subtree(tree, args_dict = {'generators' : {}}):
            def tree_gen():
                if tree.is_terminal():
                    tree2 = deepcopy(tree)
                    tree2.label_nodes()
                    tree2.make_nx_tree()
                    yield tree2
                else:
                    for children in product(*[args_dict['generators'][child.ID] for child in tree.children]):
                        gen = resolve_features(grammars[tree.language], morphology_dicts[tree.language], tree.label, [child.label for child in children])
                        for (parent_label, children_labels) in gen:
                            tree2 = deepcopy(tree)
                            tree2.children = deepcopy(children)
                            tree2.label = parent_label
                            for i in xrange(len(tree2.children)):
                                tree2.children[i].label = children_labels[i]
                            tree2.label_nodes()
                            tree2.make_nx_tree()
                            yield tree2
            args_dict['generators'][tree.ID] = tree_gen()
            return tree_gen()
        return (self.postorder_traverse(resolve_features_at_subtree, {'generators' : {}}))
    def get_sentence(self):
        """Takes the leaf nodes of the tree and returns a string of their labels in sequence."""
        words = []
        for i in sorted([ind.index for ind in self.subtree_dict.keys()]):
            if isinstance(self.subtree_dict[i].label, (str, unicode)):
                words.append(self.subtree_dict[i].label)
        return ' '.join(words) 
    @classmethod
    def from_nltk_tree(cls, T, language):
        if isinstance(T, nltk.tree.Tree):
            return cls(T.label(), [cls.from_nltk_tree(child, language) for child in T], 0, language)
        return cls(T, [], 0, language)

def adjust_tree(tree, args_dict = {}):
    """Does some syntactic operations for Japanese trees. Inserts vacuous subjects whenever T' projects to TP, inserts an ambiguous determiner whenever a common noun occurs as a DP, and gives predicate DPs a dummy "particle" in order to extract them."""
    if ((tree.category() == 'VBar') and (len(tree.children) == 2) and (tree.children[1].label.has_key('SUBCAT')) and (tree.children[1].label['SUBCAT'] == 'copula')):
        if (tree.children[0].label[feature_type] == 'DP'):
            DP = tree.children[0].label
            tree.children[0].label = FeatStructNonterminal(dict([item for item in DP.items() if (item[0] != 'PARTICLE')] + [('PARTICLE', 'pred')]))  # give the DP a dummy particle
    if ((tree.category() == 'TP') and (len(tree.children) == 1)):  # insert vacuous subject node
        tree.children = [SynTree(Trace(tree.children[0].ID, False), [], tree.QR_level, tree.language), tree.children[0]]
    if ((tree.category() == 'DBar') and (len(tree.children) == 1) and (tree.children[0].category() == 'NP')):  # insert ambiguous determiner
        tree.children = [SynTree(FeatStructNonterminal([('PropN', False), (feature_type, 'D'), ('TRACE', False)]), [SynTree('*det*', [], tree.QR_level, tree.language)], tree.QR_level, tree.language), tree.children[0]]
    return args_dict

def raise_DPs(tree, args_dict = {'DPs' : []}):
    """Raises DPs to QR position replacing them with coreferential traces. (Note: assumes that no DP is descended from another DP)."""
    tree.QR_level += 1
    is_QR_DP = (tree.category() == 'DP') and (not tree.label['TRACE'])
    if (not tree.is_terminal() and tree.label.has_key('PARTICLE')):
        is_QR_DP &= (tree.label['PARTICLE'] != 'none')  # NB: This is a stopgap for something more refined
    if is_QR_DP:
        DP_copy = deepcopy(tree)
        args_dict['DPs'].append(DP_copy)
        label = dict((key, tree.label[key]) for key in tree.label.keys())
        label['TRACE'] = True
        tree.label = FeatStructNonterminal(label)
        tree.children = [SynTree(Trace(DP_copy.ID), [], tree.QR_level, tree.language)]  # ID is coreferential with the ID of the copy
        tree.make_nx_tree()
    return args_dict

def remove_QRed_DPs(tree):
    """Strips away any DPs in QR position, leaving just a bare sentence."""
    if (len(tree.children) > 1):
        if isinstance(tree.children[0].label, Index):
            return remove_QRed_DPs(tree.children[1])
        if (isinstance(tree.children[1].label, FeatStructNonterminal) and (tree.children[1].label[feature_type] == 'PA')):
            return remove_QRed_DPs(tree.children[1])
    return tree


###########################
# SEMANTIC INTERPRETATION #
###########################

class InterpretationError(Exception):
    pass

def type_is_arg_of(type1, type2):
    """Returns True if type 1 is the type of type2's argument."""
    if (not isinstance(type2, ComplexType)):
        return False
    return (type1 == type2.first)

def interpret_tree(tree, args_dict = {'lexicon' : english_lexicon, 'generators' : {}}):
    """Gets the semantic denotation of a syntactic constituent (tree node). Does this recursively. If the tree is a leaf node, fetches the denotation from the lexicon. Otherwise applies one of four rules: 1) Projection: When there is only one child, the parent just takes the denotation of the child; 2) Function Application, in which one child takes the other as an argument; 3) Predicate Modification, where both children are of type <e, t>, and the parent's denotation is the intersection of the two children; 4) Predicate Abstraction, where one child is of type t with a free variable, the other child is a binder, and the parent is the resulting bound lambda expression. For each node in the tree, saves a generator enumerating all the (type-consistent) denotations for any of the subtrees."""
    if (len(tree.children) > 2):
        raise InterpretationError("Nodes must be at most binary branching.")
    assert(len(tree.children) <= 2)
    def tree_gen():
        if tree.is_terminal():
            if isinstance(tree.label, Trace):  # a trace, hence a variable expression
                tree2 = deepcopy(tree)
                tree2.denotation = VariableExpression(Variable("x%s" % tree.label.index))
                tree2.rule = "TRACE" if tree.label.bound else "FREEVAR"
                tree2.label_nodes()
                yield tree2
            elif isinstance(tree.label, Index):  # a binder -- represent this as a variable
                tree2 = deepcopy(tree)
                tree2.denotation = Variable("x%s" % tree.label)
                tree2.rule = "BINDER"
                tree2.label_nodes()
                yield tree2
            else:  # a lexical entry -- look up denotation in the lexicon
                if (tree.label not in args_dict['lexicon']):
                    raise InterpretationError("Could not find %s in lexicon." % tree.label)
                for cat in args_dict['lexicon'][tree.label]:
                    for denotation in args_dict['lexicon'][tree.label][cat]:
                        tree2 = deepcopy(tree)
                        tree2.denotation = denotation
                        tree2.rule = "LOOKUP"
                        tree2.label_nodes()
                        yield tree2
        elif (len(tree.children) == 1):  # Projection of meaning from child to parent
            for child in args_dict['generators'][tree.children[0].ID]:
                tree2 = deepcopy(tree)
                tree2.children = [child]
                tree2.denotation = child.denotation
                tree2.rule = "PROJ"
                tree2.label_nodes()
                yield tree2
        else:  # Binary branching
            predicate_type = Type.fromstring('<e, t>')
            x = Variable('x')
            x_expr = VariableExpression(x)
            x_expr.typecheck(signature = {'x' : 'e'})
            for (child1, child2) in product(args_dict['generators'][tree.children[0].ID], args_dict['generators'][tree.children[1].ID]):
                tree2 = deepcopy(tree)
                tree2.children = [child1, child2]
                den1, den2 = child1.denotation, child2.denotation
                if (isinstance(den1, Variable) or isinstance(den2, Variable)):  # Predicate Abstraction
                    var = den1 if isinstance(den1, Variable) else den2
                    expr = den1 if isinstance(den2, Variable) else den2
                    if (expr.type == TruthValueType()):
                        tree2.denotation = LambdaExpression(var, expr)
                        tree2.denotation.typecheck(signature = {str(var) : 'e'})
                        tree2.rule = "PA"
                        tree2.label_nodes()
                        yield tree2
                    else:  # type mismatch
                        continue
                else:
                    type1, type2 = den1.type, den2.type
                    if ((type1 == predicate_type) and (type2 == predicate_type)):  # Predicate Modification
                        xn = den1.variable
                        xn_expr = x_expr.replace(x, VariableExpression(xn), replace_bound = True)
                        tree2.denotation = LambdaExpression(xn, AndExpression(den1.applyto(xn_expr).simplify(), den2.applyto(xn_expr).simplify()))
                        tree2.rule = "PM"
                        tree2.label_nodes()
                        yield tree2
                    elif (type_is_arg_of(type1, type2) or type_is_arg_of(type2, type1)):  # Function Application
                        if type_is_arg_of(type1, type2):
                            arg_expr, func_expr = den1, den2
                        else:
                            arg_expr, func_expr = den2, den1
                        tree2.denotation = func_expr.applyto(arg_expr).simplify()
                        if (arg_expr.type == predicate_type):  # substitute the original variable name
                            tree2.denotation = tree2.denotation.replace(x, VariableExpression(arg_expr.variable), replace_bound = True)
                        tree2.rule = "FA"
                        tree2.label_nodes()
                        yield tree2
                    elif (set([type1, type2]) == set([EntityType(), TruthValueType()])):  # Topic Substitution
                        ent = den1 if (type1 == EntityType()) else den2
                        expr = den1 if (type1 == TruthValueType()) else den2
                        for var in expr.free():
                            tree3 = deepcopy(tree2)
                            tree3.denotation = expr.replace(var, ent)
                            tree3.rule = "TS"
                            tree3.label_nodes()
                            yield tree3
                    else:  # type mismatch
                        continue
    args_dict['generators'][tree.ID] = tree_gen()
    return tree_gen()



#######################
# SENTENCE GENERATION #
#######################

def split_conjuncts(expr):
    """Takes an AndExpression, which must be binary, and splits it into all its conjuncts."""
    if isinstance(expr, AndExpression):
        conjuncts = split_conjuncts(expr.first) + split_conjuncts(expr.second)
    else:
        conjuncts = [expr]
    return conjuncts

def join_conjuncts(conjuncts):
    """Takes a list of Expressions, and conjoins them (recursively) into an AndExpression."""
    if (len(conjuncts) == 0):
        return EmptyExpression()
    elif (len(conjuncts) == 1):
        return conjuncts[0]
    return AndExpression(conjuncts[0], join_conjuncts(conjuncts[1:]))

def powerset(xs):
    """Iterator over sublists of a list. The order of cardinalities is n - 1, n - 2, ..., 1, 0, n, which seems to be roughly the likelihood order for number of subject terms in a subject-predicate construction."""
    cards = list(reversed(xrange(len(xs)))) + [len(xs)]
    return list(chain.from_iterable(combinations(xs, n) for n in cards))

def diff(xs, ys):
    """Returns the list xs - ys."""
    return [x for x in xs if x not in ys]

def _str_as_subex(expr, is_subex):
    return "(%s)" % str(self) if is_subex else str(self)

class NLQuantifiedExpression(QuantifiedExpression):
    def __init__(self, variable, restrictor, nucleus):
        """Natural language quantificational expression. Takes bound variable, restrictor predicates, and nuclear predicates. Note: This subclass is for convenience, and it is not complete."""
        super(NLQuantifiedExpression, self).__init__(variable, nucleus)
        self.restrictor = restrictor
        self.nucleus = self.term
    def __str__(self):
        def _str_as_subex(expr):
            return str(expr) if (isinstance(expr, (ApplicationExpression, EmptyExpression)) or ([str(expr)[0], str(expr)[-1]] == ['(', ')'])) else "(%s)" % str(expr)
        if isinstance(self.restrictor, EmptyExpression):
            return ("%s %s . %s" % (self.getQuantifier(), self.variable, _str_as_subex(self.nucleus)))
        return ("%s %s | %s . %s" % (self.getQuantifier(), self.variable, _str_as_subex(self.restrictor), _str_as_subex(self.nucleus)))
    def __repr__(self):
        return '<' + self.__class__.__name__ + ': ' + str(self) + '>'

class EveryExpression(NLQuantifiedExpression):
    def getQuantifier(self):
        return "EVERY"

class SomeExpression(NLQuantifiedExpression):
    def getQuantifier(self):
        return "SOME"

class TheExpression(NLQuantifiedExpression):
    def getQuantifier(self):
        return "THE"

class ThisExpression(NLQuantifiedExpression):
    def getQuantifier(self):
        return "THIS"

class NoExpression(NLQuantifiedExpression):
    def getQuantifier(self):
        return "NO"

class ProperNounExpression(NLQuantifiedExpression):
    def __init__(self, variable, pnoun, nucleus):
        """Represents a proper-noun-quantified expression. pnoun is a proper noun, e.g. 'JOHN' or 'MARY'."""
        super(ProperNounExpression, self).__init__(variable, EmptyExpression(), nucleus)
        assert(pnoun in proper_nouns)
        self.pnoun = pnoun
    def getQuantifier(self):
        return self.pnoun

class EmptyExpression(Expression):
    def __init__(self):
        pass
    def __str__(self):
        return ""
    def __repr__(self):
        return '<' + self.__class__.__name__ + '>'

def dequantify(expr):
    """Given an HOL logic expression, returns a generator of possible NL logic expressions. For now, quantifier is one of {'EVERY', 'SOME', 'THE', 'THIS', 'NO'}. restrictor is an <e, t> expression restricting the domain of subjects. nucleus is an <e, t> expression asserting a predicate of the subject."""
    if isinstance(expr, AllExpression):
        term = expr.term
        if isinstance(term, ImpExpression):
            var = expr.variable
            for (perm1, perm2) in product(permutations(split_conjuncts(term.first)), permutations(split_conjuncts(term.second))):
                for (expr1, expr2) in product(dequantify(join_conjuncts(perm1)), dequantify(join_conjuncts(perm2))):
                    yield EveryExpression(var, expr1, expr2)
    elif isinstance(expr, ExistsExpression):
        var = expr.variable
        conjuncts = split_conjuncts(expr.term)
        num_unique = [Variable('UNIQUE') in conjunct.predicates() for conjunct in conjuncts].count(True)
        if (num_unique == 0):
            indices = [i for i in xrange(len(conjuncts)) if (var in conjuncts[i].variables())]
            for subset in powerset(indices):
                for (perm1, perm2) in product(permutations([conjuncts[i] for i in subset]), permutations([conjuncts[i] for i in diff(xrange(len(conjuncts)), subset)])):
                    for (expr1, expr2) in product(dequantify(join_conjuncts(perm1)), dequantify(join_conjuncts(perm2))):    
                        yield SomeExpression(var, expr1, expr2)
        elif (num_unique == 1):
            index = [Variable('UNIQUE') in conjunct.predicates() for conjunct in conjuncts].index(True)
            unique_conjuncts = split_conjuncts(conjuncts[index].argument.term)
            if (set(unique_conjuncts).issubset(set(conjuncts))):
                num_near = [(conjunct.predicates() == set([Variable('NEAR')])) and (conjunct.variables() == set([var])) for conjunct in conjuncts].count(True)
                if (num_near == 0):
                    for (perm1, perm2) in product(permutations(unique_conjuncts), permutations([conjunct for conjunct in conjuncts if conjunct not in (unique_conjuncts + [conjuncts[index]])])):
                        for (expr1, expr2) in product(dequantify(join_conjuncts(perm1)), dequantify(join_conjuncts(perm2))):
                            yield TheExpression(var, expr1, expr2)
                elif (num_near == 1):
                    index2 = [(conjunct.predicates() == set([Variable('NEAR')])) and (conjunct.variables() == set([var])) for conjunct in conjuncts].index(True)
                    for (perm1, perm2) in product(permutations(unique_conjuncts), permutations([conjunct for conjunct in conjuncts if conjunct not in (unique_conjuncts + [conjuncts[index], conjuncts[index2]])])):
                        for (expr1, expr2) in product(dequantify(join_conjuncts(perm1)), dequantify(join_conjuncts(perm2))):
                            yield ThisExpression(var, expr1, expr2)
    elif isinstance(expr, NegatedExpression):
        term = expr.term
        if isinstance(term, ExistsExpression):  # for now, must be "no" quantifier, not a simple negated sentence
            for expr2 in dequantify(term):
                yield NoExpression(expr2.variable, expr2.restrictor, expr2.nucleus)
    elif isinstance(expr, AndExpression):
        for (first, second) in product(dequantify(expr.first), dequantify(expr.second)):
            yield AndExpression(first, second)
    elif isinstance(expr, (ApplicationExpression, EqualityExpression)):
        has_proper_noun = False
        args = reversed(expr.args) if isinstance(expr, ApplicationExpression) else [expr.first, expr.second]
        for arg in args:
            if (arg.variable.name in proper_nouns):
                has_proper_noun = True
                var = unique_variable()
                nucleus = expr.replace(arg.variable, VariableExpression(var))
                for expr1 in dequantify(nucleus):
                    yield ProperNounExpression(var, arg.variable.name, expr1)
                break
        if (not has_proper_noun):
            yield expr
    else:
        yield expr

class GenerationError(Exception):
    pass

def get_free_trace_DP(tree, args_dict = {'DPs' : []}):
    """Retrieves DP subtree representing a free trace variable."""
    is_free_trace_DP = (tree.category() == 'DP') and (tree.label['TRACE']) and (not tree.children[0].label.bound)
    if is_free_trace_DP:
        args_dict['DPs'].append(tree)
    return args_dict

def expression_to_english_NP_tree(expr):
    """expr is an NL logic expression meant to be interpreted as an NP. Attempts to generate an NP tree for it."""
    if isinstance(expr, ApplicationExpression):
        preds = list(expr.predicates())
        if ((len(preds) != 1) or (preds[0].name not in common_nouns)):  # for now, only common nouns allowed
            raise GenerationError("Head of restrictor predicate must be common noun.")    
        return SynTree(deepcopy(default_featstructs['NP']), [SynTree(deepcopy(default_featstructs['NBar']), [SynTree(default_featstructs['N'], [SynTree(preds[0].name, [])])])])
    elif isinstance(expr, AndExpression):
        adj_preds = list(expr.first.predicates())
        if ((len(adj_preds) != 1) or (adj_preds[0].name not in adjectives)):  # for now, only adjectives can pre-modify NPs
            raise GenerationError("Modifier of NP must be an adjective.")
        NP_tree = expression_to_english_NP_tree(expr.second)
        adj_subtree = SynTree(default_featstructs['AdjP'], [SynTree(default_featstructs['AdjBar'], [SynTree(default_featstructs['Adj'], [SynTree(adj_preds[0].name, [])])])])
        NP_tree.children = [SynTree(deepcopy(NP_tree.children[0].label), [adj_subtree, NP_tree.children[0]])]
        return NP_tree
    else:
        raise GenerationError("Invalid NP expression.")

def expression_to_english_DP_tree(quantifier, expr = None):
    """Attempts to generate a DP tree. quantifier is either a proper noun or quantifier determiner. If the latter, expr is a proper noun (string) or an NL logic expression meant to be interpreted as an NP."""
    if (quantifier in proper_nouns):
        return SynTree(default_featstructs['DP_PropN'], [SynTree(default_featstructs['DBar_PropN'], [SynTree(default_featstructs['D_PropN'], [SynTree(quantifier, [])])])])
    elif (quantifier in quantifiers):
        NP_tree = expression_to_english_NP_tree(expr)
        return SynTree(default_featstructs['DP_quant'], [SynTree(default_featstructs['DBar_quant'], [SynTree(default_featstructs['D_quant'], [SynTree(quantifier, [])]), NP_tree])])
    else:
        raise GenerationError("Invalid quantifier '%s'." % quantifier)

def expression_to_english_tree(expr):
    """Given a sentential NL logic expression (i.e. an output of dequantify generator), attempts to generate an English syntax tree for it."""
    if isinstance(expr, ApplicationExpression):
        pred_name = expr.pred.variable.name
        if (not (pred_name in (adjectives + intransitive_verbs + transitive_verbs))):
            raise GenerationError("Invalid predicate: %s" % pred_name)
        # might want to add a line enforcing variable name to begin with an x, y, or z?
        freevars = [Trace(Index(int(arg.variable.name[1:])), False) for arg in expr.args]
        if (pred_name in adjectives):
            tree = SynTree(default_featstructs['TP'], [SynTree(default_featstructs['DP_trace'], [SynTree(freevars[0], [])]), SynTree(default_featstructs['TBar'], [SynTree(default_featstructs['VP'], [SynTree(default_featstructs['VBar'], [SynTree(default_featstructs['V_link'], [SynTree('BE', [])]), SynTree(default_featstructs['AdjP'], [SynTree(default_featstructs['AdjBar'], [SynTree(default_featstructs['Adj'], [SynTree(pred_name, [])])])])])])])])
        elif (pred_name in intransitive_verbs):
            tree = SynTree(default_featstructs['TP'], [SynTree(default_featstructs['DP_trace'], [SynTree(freevars[0], [])]), SynTree(default_featstructs['TBar'], [SynTree(default_featstructs['VP'], [SynTree(default_featstructs['VBar'], [SynTree(default_featstructs['V_intrans'], [SynTree(pred_name, [])])])])])])
        else:
            tree = SynTree(default_featstructs['TP'], [SynTree(default_featstructs['DP_trace'], [SynTree(freevars[1], [])]), SynTree(default_featstructs['TBar'], [SynTree(default_featstructs['VP'], [SynTree(default_featstructs['VBar'], [SynTree(default_featstructs['V_trans'], [SynTree(pred_name, [])]), SynTree(default_featstructs['DP_trace'], [SynTree(freevars[0], [])])])])])])
    elif isinstance(expr, EqualityExpression):
        freevars = [Trace(Index(int(arg.variable.name[1:])), False) for arg in [expr.first, expr.second]]
        tree = SynTree(default_featstructs['TP'], [SynTree(default_featstructs['DP_trace'], [SynTree(freevars[0], [])]), SynTree(default_featstructs['TBar'], [SynTree(default_featstructs['VP'], [SynTree(default_featstructs['VBar'], [SynTree(default_featstructs['V_link'], [SynTree('BE', [])]), SynTree(default_featstructs['DP_trace'], [SynTree(freevars[1], [])])])])])])
    elif isinstance(expr, NLQuantifiedExpression):
        nucleus_tree = expression_to_english_tree(expr.nucleus)
        DPs = nucleus_tree.postorder_traverse(get_free_trace_DP, {'DPs' : []})['DPs']
        if (len(DPs) == 0):
            raise GenerationError("Quantifier %s must bind a free variable." % expr.getQuantifier())
        subj_tree = expression_to_english_DP_tree(expr.getQuantifier(), expr.restrictor)
        tree = SynTree(default_featstructs['TP'], [subj_tree, SynTree(default_featstructs['PA'], [SynTree(Index(-1), []), nucleus_tree])])
        for DP in DPs:
            if (int(expr.variable.name[1:]) == DP.children[0].label.index.index):  # check that variables match
                tree[1][0].label = DP.children[0].label.index = tree[0].ID
                DP.children[0].label.bound = True  # quantifier has bound the free variable
                break
        if (tree[1][0].label.index == -1):
            raise GenerationError("Quantifier %s failed to find corresponding free variable." % expr.getQuantifier())
    else:
        raise GenerationError("Invalid expression.")
    tree.set_QR_level(1)
    tree.label_nodes()
    tree.make_nx_tree()
    return tree


##############
# MORPHOLOGY #
##############

def share_features(feat1, feat2):
    """Given two FeatStructs that match, returns the pair with any variable features filled in. If the pair doesn't match, returns None."""
    if (feat1 == feat2):  # in case entries are strings
        return (deepcopy(feat1), deepcopy(feat2))
    d1, d2 = dict(feat1.items()), dict(feat2.items())
    for key in set(d1.keys()).intersection(set(d2.keys())):
        if ((not isinstance(d1[key], Variable)) and (not isinstance(d2[key], Variable)) and (d1[key] != d2[key])):
            return None
        if (isinstance(d1[key], Variable) and (not isinstance(d2[key], Variable))):
            d1[key] = d2[key]
        if (isinstance(d2[key], Variable) and (not isinstance(d1[key], Variable))):
            d2[key] = d1[key]
    return (FeatStructNonterminal(d1), FeatStructNonterminal(d2))

def resolve_features(grammar, mdict, parent, children):
    """Given grammar, parent FeatStruct, and list of children FeatStructs (or strings), attempts to find a rule in the grammar in which the parent can generate the children. If no rule can be found, returns empty generator. Otherwise, returns generator yielding (parent, children) where any unknown features are resolved. If parent has a single child that is a string, attempts to find suitable entries in the morphology dictionary mdict."""
    if ((len(children) == 1) and (isinstance(children[0], (str, unicode)))):  # case where child is a lexeme
        for featstruct in mdict[children[0]]:
            pair = share_features(parent, featstruct)
            if pair:
                for word in mdict[children[0]][featstruct]:
                    yield (pair[0], [word])
    elif ((len(children) in [1, 2]) and all([isinstance(child, FeatStructNonterminal) for child in children])):
        for prod in grammar.productions():
            parent2, children2 = prod.lhs(), list(prod.rhs())  # from the production rule
            parent_pair = share_features(parent, parent2)
            if parent_pair:
                if (len(children) != len(children2)):  # can't be a valid production rule
                    continue
                child_pairs = [share_features(children[i], children2[i]) for i in xrange(len(children))]
                if all(child_pairs):
                    var_dict = dict()  # maps (feature, variable) to structures, e.g. [True, [False, True]] means that it occurs in the parent and the right child
                    for feature, val in parent2.items():
                        if isinstance(val, Variable):
                            var_dict[(feature, val)] = [True, [False for child in children2]]
                    for i in xrange(len(children2)):
                        for feature, val in children2[i].items():
                            if isinstance(val, Variable):
                                if var_dict.has_key((feature, val)):
                                    var_dict[(feature, val)][1][i] = True
                                else:
                                    var_dict[(feature, val)] = [False, [(i == j) for j in xrange(len(children2))]]
                    features = set([key[0] for key in var_dict.keys()])
                    variables = set([key[1] for key in var_dict.keys()])
                    if ((len(features) != len(var_dict)) or (len(variables) != len(var_dict))):
                        raise ValueError("Not a one-to-one mapping between variables and features.")
                    var_val_dict = dict()
                    mismatch = False
                    for (feature, var), val in var_dict.items():
                        values_of_var = []
                        if (val[0] and (parent[feature] != var)):
                            values_of_var.append(parent[feature])
                        for i in xrange(len(children)):
                            if (val[1][i] and (children[i][feature] != var)):
                                values_of_var.append(children[i][feature])
                        if (len(set(values_of_var)) > 1):  
                            mismatch = True
                            break
                        if (len(values_of_var) > 0):
                            var_val_dict[var] = values_of_var[0]
                    if mismatch:
                        continue
                    d_parent = dict(parent_pair[0].items())
                    d_children = [dict(child_pair[0].items()) for child_pair in child_pairs]
                    for (var, val) in var_val_dict.items():
                        for d in ([d_parent] + d_children):
                            for feature in d:
                                if (d[feature] == var):
                                    d[feature] = val
                    yield (FeatStructNonterminal(d_parent), [FeatStructNonterminal(d_child) for d_child in d_children])
      

#############
# Load CFGs #
#############

e_parser = nltk.load_parser('english.fcfg', trace = 0)
j_parser = nltk.load_parser('japanese.fcfg', trace = 0)
grammars = {'english' : e_parser.grammar(), 'japanese' : j_parser.grammar()}
morphology_dicts = {'english' : morphology_dict}

def parse(sent, parser, language = 'english'):
    tokens = sent.split()
    trees = parser.parse(tokens)
    tree = SynTree.from_nltk_tree(trees.next(), language)
    return tree

def e_parse(sent):
    return parse(sent, e_parser, 'english')

def j_parse(sent):
    return parse(sent, j_parser, 'japanese')


#########
# TESTS #
#########

num_sents = 8

e_sents = []
e_sents.append("John runs")
e_sents.append("John saw Mary")
e_sents.append("John likes Mary")
e_sents.append("John is a farmer")
e_sents.append("I am late")
e_sents.append("all farmers are happy")
e_sents.append("every farmer has a donkey")
e_sents.append("every busy farmer likes a happy healthy donkey")

e_trees = [e_parse(sent) for sent in e_sents]
e_trees2 = [list(e_trees[i].interpret())[0] for i in xrange(num_sents)]
e_trees3 = [expression_to_english_tree(dequantify(e_trees2[i].denotation).next()).quantifier_lower() for i in xrange(num_sents)]

j_sents = []
j_sents.append("John wa hashirimasu")
j_sents.append("John wa Mary wo mimashita")
j_sents.append("John wa Mary ga suki desu")
j_sents.append("John wa nouka desu")
j_sents.append("watashi wa okureteimasu")
j_sents.append("subete-no nouka wa ureshii desu")
j_sents.append("subete-no nouka wa roba wo motteimasu")
j_sents.append("subete-no nigiyaka na nouka wa ureshii genki na roba ga suki desu")  # correct up to morphology

j_trees = [j_parse(sent) for sent in j_sents]
j_trees2 = [list(j_trees[i].interpret())[0] for i in xrange(num_sents)]

for i in xrange(num_sents):  # ensure logical equivalence of the sentence pairs
    assert(e_trees2[i].denotation == j_trees2[i].denotation)


