import collections
import itertools
import nltk.metrics
import os
import random

from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.classify import NaiveBayesClassifier

import stopwatch

# Time the script execution
time = stopwatch.Timer()

def pos_word_feats(words):
    return dict([(word, True) for word in words
                 if word in top_pw])

def neg_word_feats(words):
    return dict([(word, True) for word in words
                 if word in top_nw])

# Set the working folder to the nltk_data corpora folder
os.chdir ('c:/Tools/nltk_data/corpora')

reader = CategorizedPlaintextCorpusReader('./cyberbullying_13',
                                          r'.*\.txt',
                                          cat_pattern=r'(\w+)/*')

# The questions containing bullying are the positive tests
pos_ids = reader.fileids('bullying')

# The questions not containing bullying are the negative tests
neg_ids = reader.fileids('not_bullying')

pos_words = []
neg_words = []

for fileid in pos_ids:
    pos_words += [word for word in (reader.words(fileid))]

for fileid in neg_ids:
    neg_words += [word for word in (reader.words(fileid))]
    
pw_dist =  nltk.FreqDist(pos_words)
nw_dist =  nltk.FreqDist(neg_words)

top_pw = pw_dist.keys()[:int(len(pw_dist.keys())*.10)]
top_nw = nw_dist.keys()[:int(len(nw_dist.keys())*.10)]

pos_feat = [(pos_word_feats(reader.words(fileids=[f])),
             'bullying')
            for f in pos_ids]

neg_feat = [(neg_word_feats(reader.words(fileids=[f])),
             'not_bullying')
            for f in neg_ids]

...
