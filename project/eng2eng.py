# Translate a simple English sentence (constrained to the lexicon in lexicon.py) back into English
# If -o flag is set, generates tree diagrams of various stages in the translation
# Example: python eng2eng.py "John saw Mary" -v -o saw_eng_

import sys
import argparse
from sem import *


def english_to_english(e_sent, verbose = True, img_filename = None):
    """Returns a generator of English translations of an English sentence."""
    filenames = [None for i in xrange(4)] if (img_filename is None) else [img_filename + str(i + 1) for i in xrange(4)]
    def translation_gen():
        if verbose:
            print("\nINPUT ENGLISH SENTENCE")
            print(e_sent)
        parse_tree = e_parse(e_sent)
        if verbose:
            print("\nENGLISH PARSE TREE")
            print(parse_tree)
            parse_tree.draw(filename = filenames[0])
        interpretation_gen = parse_tree.interpret()
        for interpreted_tree in interpretation_gen:
            if verbose:
                print("\nPOST-MOVEMENT TREE")
                print(interpreted_tree)
                print("\nHIGHER-ORDER LOGIC INTERPRETATION")
                print(interpreted_tree.denotation)
                interpreted_tree.draw(filename = filenames[1])
            NL_logic_expr_gen = dequantify(interpreted_tree.denotation)
            for NL_logic_expr in NL_logic_expr_gen:
                try:
                    eng_tree_no_morphology = expression_to_english_tree(NL_logic_expr).quantifier_lower()
                    if verbose:
                        print("\nNL LOGIC INTERPRETATION")
                        print(NL_logic_expr)
                        print("\nOUTPUT ENGLISH TREE (NO MORPHOLOGY)")
                        print(eng_tree_no_morphology)
                        eng_tree_no_morphology.draw(filename = filenames[2])
                except GenerationError:
                    continue
                english_tree_gen = eng_tree_no_morphology.resolve_features_bottom_up()
                for english_tree in english_tree_gen:
                    if ((parse_tree.label['TENSE'], parse_tree.label['NUM'], parse_tree.label['PERSON']) == (english_tree.label['TENSE'], english_tree.label['NUM'], english_tree.label['PERSON'])):  # tense must match the original
                        english_sentence = english_tree.get_sentence()
                        if verbose:
                            print("\nOUTPUT ENGLISH SENTENCE")
                            print(english_sentence)
                            print("")
                            english_tree.draw(filename = filenames[3])
                        yield english_sentence
    return translation_gen()

def main():
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("input")
    PARSER.add_argument("--verbose", "-v", default = False, action = 'store_true')
    PARSER.add_argument("--output", "-o", default = None, type = str)
    args = PARSER.parse_args()

    gen = english_to_english(args.input, args.verbose, args.output)
    x = gen.next()

if __name__ == "__main__":
    main()