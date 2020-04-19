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
target = []
data = []

# Read the labels and data values into lists
for line in csv_reader:
    target.append(int(line[0]))
    data.append(line[1])

# Close file
cvs_file.close()

# Create a pipeline with a transformer and a estimator
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf',   MultinomialNB()),
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
    'clf__alpha': [0.5, 0.25, 0.1, 0.75, 0.05],
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


