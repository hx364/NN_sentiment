import json
import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize
import itertools
import sys
import cPickle

sys.path.append('../../src/')

from data_prepare import *


def read_yelp(file_name='yelp_academic_dataset_review.json'):

    f = open(file_name)
    f = f.readlines()
    f = [eval(l.strip()) for l in f]
    stars = [i['stars'] for i in f]
    text = [i['text'] for i in f]

    df = pd.DataFrame()
    df['stars'] = stars
    df['text'] = text

    #compute the number of sentences in each doc
    l = list(df.text)
    text = [sent_tokenize(i) for i in list(df.text)]
    text_len = [len(i) for i in text]

    #2225188 in total
    #2089287 for length<=20
    #1654640 for length<=10
    #We decide to only consider length<=7 here
    df['length'] = text_len
    df['text_split'] = text
    return df

def filter_dataset(df, maxlen_doc=7, maxlen_sent=50, min_wc=10):
    """
    Filter the yelp dataset, by the number of sentences in a doc, and number of words in a sentence

    :param maxlen_doc: max counts of sentences in a doc
    :param maxlen_sent: max counts of words in a sent
    :param min_wc: min counts of a word in a vocabulary, delete less fequent words
    :return:
    """
    dfs = df[df.length<=maxlen_doc]

    #clean text
    text = [[clean_str(i) for i in j] for j in list(dfs.text_split)]
    text = [[s.split(" ") for s in j] for j in text]

    #get word count dictionary
    word_count = Counter()
    for doc in text:
        for sent in doc:
            word_count.update(sent)

    #233811 words in corpus
    #188590 words appear less than 10 times
    #168652 words appear less than 5 times
    #delete less frequent words
    text =[[[i for i in j if word_count[i]>=min_wc] for j in doc] for doc in text]
    dfs['text_split'] = text

    sentence_length = [max([len(sent) for sent in doc]) for doc in x]
    dfs['max_sentence_length'] = sentence_length
    dfs = dfs[dfs.max_sentence_length<=maxlen_sent]
    return dfs


def build_input(df):
    #build the word count again
    #get word count dictionary
    text = df.text_split
    word_counts = Counter()
    for doc in text:
        for sent in doc:
            word_counts.update(sent)

    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i+1 for i, x in enumerate(vocabulary_inv)}

    #get the final data
    x = [[[vocabulary[word] for word in sentence] for sentence in doc] for doc in text]
    y = list(df.stars)
    return x, y, vocabulary, vocabulary_inv

if __name__ == '__main__':
    df = read_yelp()
    df = filter_dataset()
    x, y, vocabulary, vocabulary_inv = build_input(df)

    #get the word embedding matrix
    word2vec = vocab_to_word2vec('../GoogleNews-vectors-negative300.bin', vocabulary)
    embedding_mat = build_word_embedding_mat(word2vec, vocabulary_inv)
    print '{} docs, {} words in corpus'.format(len(x), len(vocabulary_inv))

    #dump the data
    cPickle.dump([x, y, embedding_mat], open("data_x_y_w2v.p", "wb"))
    cPickle.dump([vocabulary, vocabulary_inv], open("vocabulary.p", "wb"))




