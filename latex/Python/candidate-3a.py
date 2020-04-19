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

# Read in the new csv file and create a cvs reader
new_cvs_file   = open('hold_back_01.csv') 
new_csv_reader = csv.reader(new_cvs_file)

X_test = []

for new_line in new_csv_reader:
    X_test.append(new_line[0])


# Close files
cvs_file.close()
new_cvs_file.close()

# Shuffle the neg data
random.shuffle(data_neg)
random.shuffle(data_pos)

print len(data_pos)
print len(data_neg)

# Ratio 2:1 required
multi = (len(data_neg)/2) / len(data_pos)
modul = (len(data_neg)/2) % len(data_pos)

data_pos_final = []
target_pos_final = []

for n in range(multi):
    data_pos_final = data_pos_final + data_pos
    target_pos_final = target_pos_final + target_pos

data_pos_final = data_pos_final + data_pos[:modul]
target_pos_final = target_pos_final + target_pos[:modul]

y_train = target_pos_final + target_neg
X_train = data_pos_final + data_neg

print len(X_train)
print len(X_test)

tfidf = TfidfVectorizer(
      stop_words=None
    , ngram_range=(1, 3)
    , norm='l2'
)

X_train2 = tfidf.fit_transform(X_train)
X_test2  = tfidf.transform(X_test)

model = MultinomialNB(alpha=0.3).fit(X_train2, y_train)

y_pred = model.predict(X_test2)

out_file_name = 'C:\\Users\\DColton\\Google Drive\\Masters\\MSIT - H6029 - MSc Research Project\\Data\\Corpora\\candidate_03_before.csv' 
out_file = open(out_file_name, 'w')

for n in range(len(X_test)):
    out_file.write('"')
    out_file.write(str(y_pred[n]))
    out_file.write('","')
    out_file.write(str(X_test[n]))
    out_file.write('"\n')


out_file.close()


