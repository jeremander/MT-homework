import nltk
import networkx as nx
import matplotlib.pyplot as plt
from numpy import sign
from copy import copy, deepcopy


class Index(object):
    """Simply a wrapper for an integer, so that the integer can be shared by different objects."""
    def __init__(self, index):
        if isinstance(index, Index):
            self.index = index.index
        else:
            self.index = index
    def __str__(self):
        return str(self.index)
    def __repr__(self):
        return str(self)
    def __eq__(self, other):
        if isinstance(other, Index):
            return (self.index == other.index)
        return (self.index == other)
    def __ne__(self, other):
        return (not (self == other))
    def __cmp__(self, other):
        return __cmp(self.index, other.index)
    def __hash__(self):
        return hash(self.index)


class Tree(object):
    """Native class for trees. Underlying data structure is a pair (root, subtree iterator)."""
    def __init__(self, label, children = []):
        self.label = label
        for child in children:
            assert(isinstance(child, Tree))
        self.children = children
        self.label_nodes()
    def is_terminal(self):
        """Returns True if subtree is a leaf node (has no children)."""
        return (len(self.children) == 0)
    # def preorder_traverse(self, f, args_dict = {}):
    #     """Applies function f(subtree, args_dict) to each subtree in depth-first preorder fashion. f may modify a leaf or subtree, and it may modify the args_dict so that this information can be passed to the higher nodes."""
    #     f(self, args_dict)
    #     for subtree in self:
    #         subtree.postorder_traverse(f, args_dict)
    def postorder_traverse(self, f, args_dict = {}):
        """Applies function f(subtree, args_dict) to each subtree in depth-first postorder fashion. f may modify a leaf or subtree, and it may modify the args_dict so that this information can be passed to the higher nodes."""
        for subtree in self:
            if subtree.is_terminal():
                f(subtree, args_dict)
            else:
                subtree.postorder_traverse(f, args_dict)
        return f(self, args_dict)
    def label_nodes(self):
        """Labels nodes by enumerating them in left-right depth-first order."""
        def node_count(tree, args_dict):
            if hasattr(tree, 'ID'):
                tree.ID.index = args_dict['counter']
            else:
                tree.ID = Index(args_dict['counter'])
            args_dict['subtree_dict'][tree.ID] = tree
            args_dict['counter'] += 1
        args_dict = {'counter' : 0, 'subtree_dict' : dict()}
        self.postorder_traverse(node_count, args_dict)
        self.subtree_dict = args_dict['subtree_dict']
        self.n = args_dict['counter']
    def make_nx_tree(self):
        """Create a networkx digraph of the tree, for visualization purposes."""
        self.nx_tree = nx.DiGraph()
        self.nx_tree.add_node((self.label, self.ID))
        for child in self:
            if child.is_terminal():
                nx_child = nx.DiGraph()
                nx_child.add_node((child.label, child.ID))
                root = (child.label, child.ID)
            else:
                child.make_nx_tree()
                nx_child = child.nx_tree
                root = (child.label, child.ID)
            self.nx_tree.add_nodes_from(nx_child.nodes())
            self.nx_tree.add_edges_from(nx_child.edges())
            self.nx_tree.add_edge((self.label, self.ID), root)
    def height(self):
        """Length of the longest path from root to a leaf, plus one."""
        if self.is_terminal():
            return 1
        else:
            return max([child.height() for child in self]) + 1
    def draw(self, labels = None, filename = None):
        """Via networkx digraph and graphviz package, view the tree diagram with hierarchical layout."""
        if (not hasattr(self, 'nx_tree')):
            self.make_nx_tree()
        pos = get_positions(buchheim(self))
        if (labels is None):
            labels = dict((node, node[0]) for node in self.nx_tree.nodes())
        fig = plt.figure()
        figr = fig.add_subplot(111)
        figr.axes.get_xaxis().set_visible(False)
        figr.axes.get_yaxis().set_visible(False)
        nx.draw_networkx(self.nx_tree, pos, labels = labels, node_size = 500, node_color = 'white', edge_color = 'blue', linewidths = 0, arrows = False)
        plt.show(block = False)
        if filename:
            plt.savefig(filename)
    def __eq__(self, other):
        if ((self.label != other.label) or (len(self) != len(other))):
            return False
        for i in xrange(len(self)):
            if (self[i] != other[i]):
                return False
        return True
    def __ne__(self, other):
        return (not (self == other))
    def __len__(self):
        return len(self.children)
    def __getitem__(self, i):
        return self.children[i]
    def __iter__(self):
        return (child for child in self.children)
    def __str__(self):
        s = "%s(%s, [" % (self.__class__.__name__, repr(self.label))
        for i in xrange(len(self)):
            s += repr(self[i])
            if (i < len(self) - 1):
                s += ', '
        s += "])"
        return s
    def __repr__(self):
        return str(self)
    @classmethod
    def from_nltk_tree(cls, T):
        if isinstance(T, nltk.tree.Tree):
            return cls(T.label(), [cls.from_nltk_tree(child) for child in T])
        return cls(T, [])

################
# TREE DRAWING #
################

# Code borrowed from Bill Mill's "Drawing Presentable Trees", http://github.com/llimllib/pymag-trees/

class DrawTree(object):
    def __init__(self, tree, parent = None, depth = 0, number = 1):
        self.x = self.xmin = self.xmax = -1.
        self.y = depth
        self.tree = deepcopy(tree)
        self.children = [DrawTree(c, self, depth + 1, i + 1) 
                         for i, c
                         in enumerate(self.tree.children)]
        self.parent = parent
        self.thread = None
        self.mod = 0
        self.ancestor = self
        self.change = self.shift = 0
        self._lmost_sibling = None
        #this is the number of the node in its group of siblings 1..n
        self.number = number
    def left(self): 
        return self.thread or len(self.children) and self.children[0]
    def right(self):
        return self.thread or len(self.children) and self.children[-1]
    def lbrother(self):
        n = None
        if self.parent:
            for node in self.parent.children:
                if node == self: return n
                else:            n = node
        return n
    def get_lmost_sibling(self):
        if ((not self._lmost_sibling) and self.parent and (self != self.parent.children[0])):
            self._lmost_sibling = self.parent.children[0]
        return self._lmost_sibling
    lmost_sibling = property(get_lmost_sibling)
    def __str__(self): 
        return "%s: x = %s, mod = %s" % (self.tree, self.x, self.mod)
    def __repr__(self): 
        return self.__str__()

def buchheim(tree):
    dt = firstwalk(DrawTree(tree))
    min = second_walk(dt)
    return dt

def firstwalk(v, distance = 1.):
    if len(v.children) == 0:
        if v.lmost_sibling:
            v.x = v.xmin = v.xmax = v.lbrother().xmax + distance
        else:
            v.x = v.xmin = v.xmax = 0.
    else:
        default_ancestor = v.children[0]
        for w in v.children:
            firstwalk(w)
            default_ancestor = apportion(w, default_ancestor, distance)
        #print "finished v =", v.tree, "children"
        execute_shifts(v)
        v.x = (v.children[0].x + v.children[-1].x) / 2
        w = v.lbrother()
        if w:
            shift = max(0., w.xmax - v.xmin) + distance  # ensure that linear precedence order is preserved
            third_walk(v, shift)  # potentially makes the algorithm O(n^2), but oh well...
        v.xmin = v.children[0].xmin
        v.xmax = v.children[-1].xmax
    return v

def second_walk(v, m = 0, depth=  0, min = None):
    v.x += m
    v.y = depth
    if min is None or v.x < min:
        min = v.x
    for w in v.children:
        min = second_walk(w, m + v.mod, depth+1, min)
    return min

def third_walk(dt, n):
    """Shifts entire tree right by n."""
    dt.x += n
    dt.xmin += n
    dt.xmax += n
    for c in dt.children:
        third_walk(c, n)

def apportion(v, default_ancestor, distance):
    w = v.lbrother()
    if w is not None:
        #in buchheim notation:
        #i == inner; o == outer; r == right; l == left; r = +; l = -
        vir = vor = v
        vil = w
        vol = v.lmost_sibling
        sir = sor = v.mod
        sil = vil.mod
        sol = vol.mod
        while vil.right() and vir.left():
            vil = vil.right()
            vir = vir.left()
            vol = vol.left()
            vor = vor.right()
            vor.ancestor = v
            shift = (vil.x + sil) - (vir.x + sir) + distance
            if shift > 0:
                move_subtree(ancestor(vil, v, default_ancestor), v, shift)
                sir = sir + shift
                sor = sor + shift
            sil += vil.mod
            sir += vir.mod
            sol += vol.mod
            sor += vor.mod
        if vil.right() and not vor.right():
            vor.thread = vil.right()
            vor.mod += sil - sor
        else:
            if vir.left() and not vol.left():
                vol.thread = vir.left()
                vol.mod += sir - sol
            default_ancestor = v
    return default_ancestor

def move_subtree(wl, wr, shift):
    subtrees = wr.number - wl.number
    #print wl.tree, "is conflicted with", wr.tree, 'moving', subtrees, 'shift', shift
    #print wl, wr, wr.number, wl.number, shift, subtrees, shift/subtrees
    wr.change -= shift / subtrees
    wr.shift += shift
    wl.change += shift / subtrees
    wr.x += shift
    wr.mod += shift

def execute_shifts(v):
    shift = change = 0.
    for w in v.children[::-1]:
        #print "shift:", w, shift, w.change
        w.x += shift
        w.mod += shift
        change += w.change
        shift += w.shift + change

def ancestor(vil, v, default_ancestor):
    #the relevant text is at the bottom of page 7 of
    #"Improving Walker's Algorithm to Run in Linear Time" by Buchheim et al, (2002)
    #http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.8757&rep=rep1&type=pdf
    if vil.ancestor in v.parent.children:
        return vil.ancestor
    else:
        return default_ancestor

def get_positions(dt, pos = dict()):
    pos[(dt.tree.label, dt.tree.ID)] = (dt.x, -dt.y)
    if (len(dt.children) > 0):
        get_positions(dt.left())
    if (len(dt.children) > 1):
        get_positions(dt.right())
    return pos


