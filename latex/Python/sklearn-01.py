import csv
import numpy as np
import os
import random

from operator import itemgetter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report

# Change folder to where the csv files are located
os.chdir('R:\\Corpora')

# Read in the csv file and create a cvs reader
cvs_file   = open('cyberbullying_01.csv') 
csv_reader = csv.reader(cvs_file)

# Create empty lists for the labels and the data
pos_labels = []
pos_data = []
neg_labels = []
neg_data = []

# Read the labels and data values into lists
for line in csv_reader:
    if int(line[0]) == 0:
        pos_labels.append(int(line[0]))
        pos_data.append(line[1])
    else:
        neg_labels.append(int(line[0]))
        neg_data.append(line[1])
        
# Close file
cvs_file.close()

# Randomise the samples
random.shuffle(pos_data)
random.shuffle(neg_data)

# Initially choose a static training set size
pos_train_size = int(round(len(pos_data)*0.80)) 
print 'Positive training dataset: ' + str(pos_train_size) + '\n'
neg_train_size = int(round(len(neg_data)*0.80)) 
print 'Negative training dataset: ' + str(neg_train_size) + '\n'

# Manipulate data into sk-learn expected format
train_data = pos_data[:pos_train_size] + neg_data[:neg_train_size]
test_data  = pos_data[pos_train_size:] + neg_data[neg_train_size:]

train_labels = pos_labels[:pos_train_size] + neg_labels[:neg_train_size]
test_labels  = pos_labels[pos_train_size:] + neg_labels[neg_train_size:]

X_train = np.array([''.join(el) for el in train_data])
y_train = np.array([el for el in train_labels])

X_test = np.array([''.join(el) for el in test_data]) 
y_test = np.array([el for el in test_labels]) 

vectorizer = TfidfVectorizer(min_df=2,
                             ngram_range=(1, 2),
                             stop_words='english',
                             strip_accents='unicode',
                             norm='l2')
 
test_string = unicode(pos_data[10])

print "Example string: " + test_string
print "Preprocessed string: " + vectorizer.build_preprocessor()(test_string)
print "Tokenized string:" + str(vectorizer.build_tokenizer()(test_string))
print "N-gram data string:" + str(vectorizer.build_analyzer()(test_string))
print "\n"

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

nb_classifier = MultinomialNB().fit(X_train, y_train)

y_nb_predicted = nb_classifier.predict(X_test)

print "MODEL: Multinomial Naive Bayes\n"

print 'The precision for this classifier is ' + str(metrics.precision_score(y_test, y_nb_predicted))
print 'The recall for this classifier is ' + str(metrics.recall_score(y_test, y_nb_predicted))
print 'The f1 for this classifier is ' + str(metrics.f1_score(y_test, y_nb_predicted))
print 'The accuracy for this classifier is ' + str(metrics.accuracy_score(y_test, y_nb_predicted))

print '\nHere is the classification report:'
print classification_report(y_test, y_nb_predicted)

#simple thing to do would be to up the n-grams to bigrams; try varying ngram_range from (1, 1) to (1, 2)
#we could also modify the vectorizer to stem or lemmatize
print '\nHere is the confusion matrix:'
print metrics.confusion_matrix(y_test, y_nb_predicted)

from sklearn.svm import LinearSVC

svm_classifier = LinearSVC().fit(X_train, y_train)

y_svm_predicted = svm_classifier.predict(X_test)
print "MODEL: Linear SVC\n"

print 'The precision for this classifier is ' + str(metrics.precision_score(y_test, y_svm_predicted))
print 'The recall for this classifier is ' + str(metrics.recall_score(y_test, y_svm_predicted))
print 'The f1 for this classifier is ' + str(metrics.f1_score(y_test, y_svm_predicted))
print 'The accuracy for this classifier is ' + str(metrics.accuracy_score(y_test, y_svm_predicted))

print '\nHere is the classification report:'
print classification_report(y_test, y_svm_predicted)

#simple thing to do would be to up the n-grams to bigrams; try varying ngram_range from (1, 1) to (1, 2)
#we could also modify the vectorizer to stem or lemmatize
print '\nHere is the confusion matrix:'
print metrics.confusion_matrix(y_test, y_svm_predicted)
