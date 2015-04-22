#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract dependency trees from the PDT. Outputs a line containing tokens denoting the parent and
label of each arc (where the head node is the current index). e.g.,

    1/Adv   0/ExD     1/AuxC 3/ExD
    Třikrát rychlejší než    slovo

implies arcs

    ROOT -ExD-> rychlejší
    rychlejší -Adv-> Třikrát
    rychlejší -AuxC-> než
    než -ExD-> slovo

usage: pull_tree.py /path/to/pdt/*a.gz/file
"""

import xml.etree.ElementTree as ET
import argparse
import codecs
import gzip
import sys

PARSER = argparse.ArgumentParser(description="Extract data from the PDT v2.0")
PARSER.add_argument("files", type=str, nargs='+', help="(compressed) file to read")
args = PARSER.parse_args()

sys.stdout = codecs.getwriter('utf-8')(sys.stdout) 

def p(key):
    """Returns PDT namespace prefix"""
    return '{http://ufal.mff.cuni.cz/pdt/pml/}%s' % (key)

def get_links(node, indices, parent = 0):
    index = int(node.findall(p('ord'))[0].text)
    if index > 0:
        label = node.findall(p('afun'))[0].text
        indices[index] = (parent,label)
    if len(node.findall(p('children'))) > 0:
        for child in node.findall(p('children'))[0].findall(p('LM')):
            get_links(child, indices, index)

for file in args.files:
    tree = ET.parse(gzip.open(file))
    trees = tree.getroot().findall(p('trees'))[0]
    for tree in trees.findall(p('LM')):
        indices = {}
        get_links(tree, indices)

        for edge in [indices[x] for x in range(1, len(indices) + 1)]:
            print '%s/%s' % (edge[0], edge[1]),
        print
