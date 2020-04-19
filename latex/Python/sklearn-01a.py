import csv
import numpy as np
import os
import random
import stopwatch

# from operator import itemgetter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import classification_report

# Time the script execution
time = stopwatch.Timer()

# Change folder to where the csv files are located
os.chdir('R:\\Corpora')

# Read in the csv file and create a cvs reader
cvs_file   = open('cyberbullying_13.csv') 
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

# Loop 5 times to get an average representation
for count in range(5):        

    X_train, X_test, y_train, y_test = \
             cross_validation.train_test_split(
                 data, target, test_size=0.2,
                 random_state=(random.randrange(1,100)))

    tfidf = TfidfVectorizer()

    X_train = tfidf.fit_transform(X_train)
    X_test  = tfidf.transform(X_test)

    # model = MultinomialNB().fit(X_train, y_train)
    model = LinearSVC().fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print 'Model Accuracy: ' \
          + str(metrics.accuracy_score(y_test, y_pred))

    print 'Classification Report:'
    print classification_report(y_test, y_pred)

    print 'Confusion Matrix:'
    print metrics.confusion_matrix(y_test, y_pred)

time.stop()
print 'Total Time:\t',     time.elapsed


