import csv
import numpy as np
import os
import random

# from operator import itemgetter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

from time import time

# Change folder to where the csv files are located
os.chdir('R:\\Corpora')

# Read in the csv file and create a cvs reader
cvs_file   = open('cyberbullying_01.csv') 
csv_reader = csv.reader(cvs_file)

# Create empty lists for the labels and the data
target_pos = []
data_pos = []
target_neg = []
data_neg = []
target = []
data = []

# Read the labels and data values into lists
for line in csv_reader:
    target_class = int(line[0])
    if target_class == 0:
        target_pos.append(target_class)
        data_pos.append(line[1])
    else:
        target_neg.append(target_class)
        data_neg.append(line[1])

# Close file
cvs_file.close()

# Shuffle the neg data
random.shuffle(data_neg)

print len(data_pos)
print len(data_neg)

target = target_pos + target_neg[:int(len(target_pos) * 1)]
data = data_pos + data_neg[:int(len(data_pos) * 1)]

print len(target)
print len(data)

# Create a pipeline with a transformer and a estimator
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf',   LinearSVC()),
])

# Define the parameter ranges
parameters = {
    'tfidf__stop_words': [None, 'english'],
    'tfidf__ngram_range': [(1, 1),
                           (1, 2),
                           (1, 3),
                           (2, 2),
                           (2, 3),
                           (3, 3)],
    'tfidf__norm': ['l1', 'l2'],
    # 'clf__alpha': [0..4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.7],
}

# Create the grid search object                           
grid_search = GridSearchCV(pipeline,
                           parameters,
                           cv=5)

# Fit the data and time
t0 = time()
grid_search.fit(data, target)
print("done in %0.3fs" % (time() - t0))

# Print the best score obtained
print("Best score: %0.3f" % grid_search.best_score_)

# Print the parameter values that gave the best score
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


