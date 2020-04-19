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
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report

import pylab as pl
from sklearn.metrics import roc_curve, auc

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

# Calculate the number of pos and neg samples
pos_sample = int(float((len(data_neg) + len(data_pos)))
                 * 0.5)
neg_sample = int(float((len(data_neg) + len(data_pos)))
                 * 0.5)

multi_pos = pos_sample / len(data_pos)
modul_pos = pos_sample % len(data_pos)

modul_neg = neg_sample % len(data_neg)

data_pos_final = []
target_pos_final = []

data_neg_final = []
target_neg_final = []

for n in range(multi_pos):
    data_pos_final = data_pos_final + data_pos
    target_pos_final = target_pos_final + target_pos

data_pos_final = data_pos_final + data_pos[:modul_pos]
target_pos_final = target_pos_final + target_pos[:modul_pos]

data_neg_final = data_neg_final + data_neg[:modul_neg]
target_neg_final = target_neg_final + target_neg[:modul_neg]

target = target_pos_final + target_neg_final
data = data_pos_final + data_neg_final

print len(data_pos_final)
print len(data_neg_final)
print len(data)

X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(
        data, target, test_size=0.2,
        random_state=(random.randrange(1,100)))

tfidf = TfidfVectorizer(
      stop_words='english'
    , ngram_range=(1, 3)
    , norm='l2'
)

X_train = tfidf.fit_transform(X_train)
X_test  = tfidf.transform(X_test)

# model = MultinomialNB(alpha=0.0075).fit(X_train, y_train)
# Run classifier
classifier = svm.SVC(kernel='linear', probability=True, random_state=0)
probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)

# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right")
pl.show()
time.stop()
print 'Total Time:\t',     time.elapsed


