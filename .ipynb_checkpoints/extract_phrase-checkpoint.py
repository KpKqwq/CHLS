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


def read_corpus(path, text_label_pair=False):
    with open(path, encoding='utf8') as f:
        examples=[line.strip() for line in f][:1]
        second_text = False
#        for i in range(len(examples)):
#            examples[i][0] = int(examples[i][0])
#            if not second_text:
#                examples[i][2] = None
#    # label, text1, text2
#    if text_label_pair:
#        tmp = list(zip(*examples))
#        return tmp[0], list(zip(tmp[1], tmp[2]))
#    else:
        return examples


# The function to extract phrases from a obtained sentence tree
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


# The core function to generate parsed phrases given a series of sentences
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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--input_file',
                           type=str,
                           required=True,
                           help='The dataset file path where the phrases need to be extracted')
    argparser.add_argument('--output_prefix',
                           type=str,
                           required=True,
                           help='The prefix string that will be added to the phrase files that are generated')
    args = argparser.parse_args()

    examples = read_corpus(args.input_file)

    # prepare parser and sentencizer
    parser = StanfordCoreNLP("/home/user/stanford-corenlp-4.3.2")
    sentencizer = English()
    
    sentencizer.add_pipe(sentencizer.create_pipe("sentencizer"))

    all_phrases = []
    all_sentences = []
    all_word_lists = []

    for example in tqdm(examples):
        text = example
        # remove the special token "%" as it will lead to error of consistency parsing
        text = text.replace("%", "")
        text = text.replace(" '", "'")
        text = text.replace("$", "")
        text = process_string(text)

        # split the whole example into multiple single sentences
        sents = sentencizer(text).sents
        local_sentences = []
        for sent in sents:
            local_sentence = sent.text
            local_sentences.append(local_sentence)

        stop_words_set = set(nltk.corpus.stopwords.words('english'))

        # phrases, local_sentences, word_lists = extract_phrases(parser, local_sentences, stop_words_set)
        phrases, local_sentences, word_lists = extract_phrases(parser, local_sentences, stop_words_set)

        all_phrases.append(phrases)
        all_sentences.append(local_sentences)
        all_word_lists.append(word_lists)
    all_phrases = np.array(all_phrases)
    all_sentences = np.array(all_sentences)
    all_word_lists = np.array(all_word_lists)
    print(all_phrases)
    print(all_sentences)
    print(all_word_lists)
    #np.save("dataset/wikilarge/parsed_data/{}_phrases.npy".format(args.output_prefix), all_phrases)
    #np.save("dataset/wikilarge/parsed_data/{}_sentences.npy".format(args.output_prefix), all_sentences)
    #np.save("dataset/wikilarge/parsed_data/{}_word_lists.npy".format(args.output_prefix), all_word_lists)
