import collections
import nltk.metrics
import os
import random

from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.classify import NaiveBayesClassifier

import stopwatch

# Time the script execution
time = stopwatch.Timer()

def word_feats(words):
    return dict([(word, True) for word in words])

# Set the working folder to the nltk_data corpora folder
os.chdir ('c:/Tools/nltk_data/corpora')

reader = CategorizedPlaintextCorpusReader('./cyberbullying_13',
                                          r'.*\.txt',
                                          cat_pattern=r'(\w+)/*')

# The questions not containing bullying are the negative tests
all_neg_ids = reader.fileids('not_bullying')
random.shuffle(all_neg_ids)

# The questions containing bullying are the positive tests
all_pos_ids = reader.fileids('bullying')
random.shuffle(all_pos_ids)

total_sample = (len(all_neg_ids) + len(all_pos_ids)) / 2

# Ratio 50:50 required
multi_pos = total_sample / len(all_pos_ids)
modul_pos = total_sample % len(all_pos_ids)

modul_neg = total_sample % len(all_neg_ids)
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

# Manually implement cross validation
# Not provided out of the box by NLTK
# Performing 10 fold cross validation

#Initialise variables to hold x-validation performance totals
pos_pre  = 0
pos_rec  = 0
pos_F    = 0
neg_pre  = 0
neg_rec  = 0
neg_F    = 0
accuracy = 0

# Variable to hold the number of folds
fold_size = 10

# Calculate the size of each fold based on the total
#   number of samples
pos_fold_size = int(len(pos_feat) / fold_size)
neg_fold_size = int(len(neg_feat) / fold_size)

# Iterate through each fold
for fold in range(fold_size):

    # The start and stop index for the pos samples
    pos_start = fold * pos_fold_size
    pos_stop  = (fold + 1) * pos_fold_size

    # The start and stop index for the neg samples
    neg_start = fold * neg_fold_size
    neg_stop  = (fold + 1) * neg_fold_size

    # Make sure the the stop index says with-in the
    #   bound of the size of the sample
    if pos_stop > len(pos_feat):
        pos_stop = len(pos_feat)
        
    if neg_stop > len(neg_feat):
        neg_stop = len(neg_feat)

    # Create the test and train datasets
    test  = pos_feat[pos_start:pos_stop] + \
            neg_feat[neg_start:neg_stop]
    train = ( (pos_feat[:pos_start] + \
               pos_feat[pos_stop:])
             + (neg_feat[:neg_start] + \
                neg_feat[neg_stop:]) )

    # Create the classifier using the train dataset
    classifier = NaiveBayesClassifier.train(train)

    # The NTLK metrics package uses sets to
    # calculate performace metrics
    # Create empty reference and test sets
    refsets  = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
 
    #Populate the metrics sets with values
    for i, (feats, label) in enumerate(test):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
     
    # Calculate precision. recall, f-measure
    #   and accuracy
    pos_pre  += nltk.metrics.precision(refsets['bullying'],
                                       testsets['bullying'])
    pos_rec  += nltk.metrics.recall(refsets['bullying'],
                                    testsets['bullying'])
    pos_F    += nltk.metrics.f_measure(refsets['bullying'],
                                       testsets['bullying'])
    neg_pre  += nltk.metrics.precision(refsets['not_bullying'],
                                       testsets['not_bullying'])
    neg_rec  += nltk.metrics.recall(refsets['not_bullying'],
                                    testsets['not_bullying'])
    neg_F    += nltk.metrics.f_measure(refsets['not_bullying'],
                                       testsets['not_bullying'])
    accuracy += nltk.classify.accuracy(classifier, test)

    # Show the top 5 most informative features
    # classifier.show_most_informative_features(5)

time.stop()

# Print out the overall performace measures
print 'pos samples:\t',    len(pos_ids)
print 'pos precision:\t',  pos_pre / fold_size
print 'pos recall:\t',     pos_rec / fold_size
print 'pos F-measure:\t',  pos_F / fold_size
print 'neg samples:\t',    len(neg_ids)
print 'neg precision:\t',  neg_pre / fold_size
print 'neg recall:\t',     neg_rec / fold_size
print 'neg F-measure:\t',  neg_F / fold_size
print 'model accuracy:\t', accuracy / fold_size
print 'Total Time:\t',     time.elapsed
