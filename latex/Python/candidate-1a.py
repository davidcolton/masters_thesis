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

def word_feats(words):
    return dict([(word, True) for word in words ])

# Set the working folder to the nltk_data corpora folder
os.chdir ('R:/corpora')

reader = CategorizedPlaintextCorpusReader('./cyberbullying_13',
                                          r'.*\.txt',
                                          cat_pattern=r'(\w+)/*')

# The questions not containing bullying are the negative tests
all_neg_ids = reader.fileids('not_bullying')
random.shuffle(all_neg_ids)

# The questions containing bullying are the positive tests
all_pos_ids = reader.fileids('bullying')
random.shuffle(all_pos_ids)

# Calculate the number of pos and neg samples
pos_sample = int(float((len(all_neg_ids) + len(all_pos_ids)))
                 * 0.4)
neg_sample = int(float((len(all_neg_ids) + len(all_pos_ids)))
                 * 0.6)

multi_pos = pos_sample / len(all_pos_ids)
modul_pos = pos_sample % len(all_pos_ids)

modul_neg = neg_sample % len(all_neg_ids)
pos_ids = []
neg_ids = []

for n in range(multi_pos):
    pos_ids = pos_ids + all_pos_ids

pos_ids = pos_ids + all_pos_ids[:modul_pos]

neg_ids = all_neg_ids[:modul_neg]

pos_feat = [(word_feats(reader.words(fileids=[f])),
             'bullying')
            for f in pos_ids]

neg_feat = [(word_feats(reader.words(fileids=[f])),
             'not_bullying')
            for f in neg_ids]

train = pos_feat + neg_feat

# Create the classifier using the train dataset
classifier = NaiveBayesClassifier.train(train)

new_reader = CategorizedPlaintextCorpusReader('./simulation_13',
                                              r'.*\.txt',
                                              cat_pattern='unknown')

new_feat = [(word_feats(new_reader.words(fileids=[f]))) for f in new_reader.fileids()]

pred = classifier.batch_classify(new_feat)

out_file_name = 'C:\\Users\\DColton\\Google Drive\\Masters\\MSIT - H6029 - MSc Research Project\\Data\\Corpora\\candidate_01__sim_before.csv' 
out_file = open(out_file_name, 'w')

for n in range(len(pred)):
    out_file.write('"')
    out_file.write(pred[n])
    out_file.write('","')
    out_file.write(' '.join(key for (key,val) in new_feat[n].iteritems()))
    out_file.write('"\n')


out_file.close()




