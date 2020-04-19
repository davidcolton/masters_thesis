import csv
import numpy as np
import os
import random
import stopwatch

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
random.shuffle(data_pos)

print len(data_pos)
print len(data_neg)

# Ratio 1:1 required
multi = (len(data_neg)/1) / len(data_pos)
modul = (len(data_neg)/1) % len(data_pos)

data_pos_final = []
target_pos_final = []

for n in range(multi):
    data_pos_final = data_pos_final + data_pos
    target_pos_final = target_pos_final + target_pos

data_pos_final = data_pos_final + data_pos[:modul]
target_pos_final = target_pos_final + target_pos[:modul]

target = target_pos_final + target_neg
data = data_pos_final + data_neg

len(data_pos)
print len(target)
print len(data)

# Loop 5 times to get an average representation
for count in range(5):        
    # Close file
    cvs_file.close()

    X_train, X_test, y_train, y_test = \
             cross_validation.train_test_split(
                 data, target, test_size=0.2,
                 random_state=(random.randrange(1,100)))

    tfidf = TfidfVectorizer(
          stop_words=None
        , ngram_range=(1, 2)
        , norm='l2'
    )

    X_train = tfidf.fit_transform(X_train)
    X_test  = tfidf.transform(X_test)

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


