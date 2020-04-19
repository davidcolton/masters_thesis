import collections
import itertools
import nltk.metrics
import os
import random

from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.classify import NaiveBayesClassifier

import stopwatch

# Time the script execution
from time import time

def word_feats(words):
    return dict([(word, True) for word in words ])

# Change folder to where the csv files are located
os.chdir('C:\\Tools\\corpora')

# Read in the simulation samples
sim_reader = CategorizedPlaintextCorpusReader('./sim_12',
                                              r'.*\.txt',
                                              cat_pattern='unknown')

sim_samples = [(word_feats(sim_reader.words(fileids=[f])))
               for f in sim_reader.fileids()]

# Shuffle the simulation data
random.shuffle(sim_samples)

# The number of simulation samples
sim_len = len(sim_samples)
sim_size = int(sim_len / 20)

# Loop 20 time
for loop in range(20):

    reader = CategorizedPlaintextCorpusReader('./cb_12',
                                              r'.*\.txt',
                                              cat_pattern=r'(\w+)/*')

    # The questions not containing bullying are the negative tests
    all_neg_ids = reader.fileids('not_bullying')
    random.shuffle(all_neg_ids)

    # The questions containing bullying are the positive tests
    all_pos_ids = reader.fileids('bullying')
    random.shuffle(all_pos_ids)

    t0 = time()

    # Calculate the number of pos and neg samples
    pos_sample = int(float((len(all_neg_ids) + len(all_pos_ids)))
                     * 0.5)
    neg_sample = int(float((len(all_neg_ids) + len(all_pos_ids)))
                     * 0.5)

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

    if (loop == 19):
        test  = sim_samples[sim_size * loop:sim_len]
    else:
        test  = sim_samples[sim_size * loop:sim_size * (loop+1)]
        
    pred = classifier.batch_classify(test)

    print (time() - t0)
    print len(train)

    for n in range(len(test)):
        file_name = 'sim_%s_%s.txt' % (str(loop), str(n))
        if pred[n] == 'bullying':
            out_file_name = '.\\cb_12\\bullying\\' + file_name
        else:
            out_file_name = '.\\cb_12\\not_bullying\\' + file_name

        out_file = open(out_file_name, 'a')
        out_file.write(' '.join(key for (key,val) in test[n].iteritems()))
        out_file.close()

    out_file_name_02 = 'sim_03.csv' 
    out_file_02 = open(out_file_name_02, 'a')

    for n in range(len(test)):
        out_file_02.write('"')
        out_file_02.write(str(pred[n]))
        out_file_02.write('","')
        out_file_02.write(' '.join(key for (key,val) in test[n].iteritems()))
        out_file_02.write('"\n')


    out_file_02.close()

# Classify hold back samples
# Read in the simulation samples
t0 = time()

# Read in the simulation samples
hb_reader = CategorizedPlaintextCorpusReader('./hb_12',
                                              r'.*\.txt',
                                              cat_pattern='unknown')

hb_samples = [(word_feats(hb_reader.words(fileids=[f])))
               for f in hb_reader.fileids()]


hb_pred = classifier.batch_classify(hb_samples)

print("done in %0.3fs" % (time() - t0))

out_hb_name = 'hb_03.csv' 
out_hb = open(out_hb_name, 'w')

for n in range(len(hb_samples)):
    out_hb.write('"')
    out_hb.write(str(hb_pred[n]))
    out_hb.write('","')
    out_hb.write(' '.join(key for (key,val) in hb_samples[n].iteritems()))
    out_hb.write('"\n')

out_hb.close()

