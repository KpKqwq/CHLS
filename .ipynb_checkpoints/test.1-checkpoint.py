import csv
import re
from string import punctuation as punc
import argparse

import nltk
import numpy as np
from nltk.tree import Tree
from spacy.lang.en import English
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
def extract_phrases(parser, local_sentences, stop_words_set):
    phrases = []
    word_lists = []
    for i, sent in enumerate(local_sentences):
        tree_str = parser.parse(sent)

        # tree_str = parser.predict(sent)["trees"]
        tree_str = add_indices_to_terminals(tree_str)
        word_list = [leave.split(",/a")[0] for leave in Tree.fromstring(tree_str).leaves()]
        for word_i, word in enumerate(word_list):
            if word == "-LRB-":
                word_list[word_i] = "("
            if word == "-RRB-":
                word_list[word_i] = ")"

        word_lists.append(word_list)
        local_sentences[i] = " ".join(word_list)
        phrases = extract_phrase(phrases, tree_str, "P", i, stop_words_set, word_list)
        # replace special tokens "-LRB" and "-RRB"
        local_sentences[i] = local_sentences[i].replace("-LRB-", "(")
        local_sentences[i] = local_sentences[i].replace("-RRB-", ")")
        for i, phrase in enumerate(phrases):
            phrases[i][0] = phrase[0].replace("-LRB-", "(")
            phrases[i][0] = phrase[0].replace("-RRB-", ")")
    return phrases, local_sentences, word_lists
def add_indices_to_terminals(tree):
    tree = Tree.fromstring(tree)
    for idx, _ in enumerate(tree.leaves()):
        tree_location = tree.leaf_treeposition(idx)
        # non_terminal = tree[tree_location[:-1]]
        # non_terminal[0] = non_terminal[0] + ",/a" + str(idx)
        non_terminal = tree[tree_location]
        tree[tree_location] = non_terminal + ",/a" + str(idx)
    return str(tree)


def convert2local(sentencizer, text):
    # change " '" to "'" such that parser can recognize it
    text = text.replace(" '", "'")

    sents = sentencizer(text).sents
    local_sentences = []
    for sent in sents:
        local_sentence = sent.text
        local_sentences.append(local_sentence)
    return local_sentences


# The function to process a string to replace some specific symbols
def process_string(string):
    string = re.sub("( )(\'[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", string)
    string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", string)
    # U . S . -> U.S.
    string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", string)
    # reduce left space
    string = re.sub("( )([,\.!?:;)])", r"\2", string)
    # reduce right space
    string = re.sub("([(])( )", r"\1", string)
    string = re.sub("s '", "s'", string)
    # reduce both space
    string = re.sub("(')( )(\S+)( )(')", r"\1\3\5", string)
    string = re.sub("(\")( )(\S+)( )(\")", r"\1\3\5", string)
    string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", string)
    string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", string)
    # string = re.sub(" ' ", "'", string)
    return string
def extract_phrase(phrases, tree_str, label, i, stop_words_set, word_list):
    trees = Tree.fromstring(tree_str)
    for tree in trees:
        for subtree in tree.subtrees():
            if len(subtree.label()) != 0:
                if subtree.label()[-1] == label:
                    # check the depth of the phrase
                    if subtree.height() <= 4:
                        t = subtree
                        tree_label = t.label()
                        leaves = t.leaves()
                        # if punctuation in the end of the phrase, delete it
                        if t.leaves()[-1].split(",/a")[0] in punc:
                            leaves = t.leaves()[:-1]
                            if len(leaves) == 0:
                                continue
                        start_index = leaves[0].split(",/a")[1]
                        end_index = leaves[-1].split(",/a")[1]
                        t = " ".join([leave.split(",/a")[0] for leave in leaves])
                        # check the stop_words
                        if t.strip().lower() not in stop_words_set:
                            phrases.append([t, tree_label, i, [int(start_index), int(end_index)]])

    return phrases

stop_words_set = set(nltk.corpus.stopwords.words('english'))
parser = StanfordCoreNLP("/home/user/stanford-corenlp-4.3.2")
local_sentences=["Admission to Tsinghua is extremely competitive ."]
print((extract_phrases(parser, local_sentences, stop_words_set))[0])
