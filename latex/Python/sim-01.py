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
import stopwatch

# Time the script execution
from time import time

# Change folder to where the csv files are located
os.chdir('R:\\Corpora')

# Read in the simulation samples
sim_file   = open('sim_01.csv') 
sim_reader = csv.reader(sim_file)

sim_samples = []

for new_line in sim_reader:
    sim_samples.append(new_line[0])

sim_file.close()

# Shuffle the simulation data
random.shuffle(sim_samples)

# The number of simulation samples
sim_len = len(sim_samples)
sim_size = int(sim_len / 20)

# Loop 20 time
for loop in range(20):

    t0 = time()

    # Reset empty lists for the labels and the data
    target_pos = []
    data_pos   = []
    target_neg = []
    data_neg   = []
    target     = []
    data       = []

    # Read in the csv file and create a cvs reader
    cvs_file   = open('cb_01.csv') 
    csv_reader = csv.reader(cvs_file)

    # Read the labels and data values into lists
    for line in csv_reader:
        target_class = int(line[0])
        if target_class == 0:
            target_pos.append(target_class)
            data_pos.append(line[1])
        else:
            target_neg.append(target_class)
            data_neg.append(line[1])

    # Close files
    cvs_file.close()

    # Shuffle the neg data
    random.shuffle(data_neg)
    random.shuffle(data_pos)

    # Ratio 2:1 required
    multi = (len(data_neg)/1) / len(data_pos)
    modul = (len(data_neg)/1) % len(data_pos)

    data_pos_final = []
    target_pos_final = []

    for n in range(multi):
        data_pos_final = data_pos_final + data_pos
        target_pos_final = target_pos_final + target_pos

    data_pos_final = data_pos_final + data_pos[:modul]
    target_pos_final = target_pos_final + target_pos[:modul]

    y_train = target_pos_final + target_neg
    X_train = data_pos_final + data_neg

    tfidf = TfidfVectorizer(
          stop_words=None
        , ngram_range=(1, 2)
        , norm='l2'
    )

    # Train Model
    X_train = tfidf.fit_transform(X_train)
    model = LinearSVC().fit(X_train, y_train)

    if (loop == 19):
        X_test  = sim_samples[sim_size * loop:sim_len]
    else:
        X_test = sim_samples[sim_size * loop:sim_size * (loop+1)]
        
    X_test2 = tfidf.transform(X_test)
    y_pred  = model.predict(X_test2)

    print (time() - t0)

    # Append classified samples to training data
    out_file_name_01 = 'cb_01.csv' 
    out_file_01 = open(out_file_name_01, 'a')

    for n in range(len(X_test)):
        out_file_01.write('"')
        out_file_01.write(str(y_pred[n]))
        out_file_01.write('","')
        out_file_01.write(str(X_test[n]))
        out_file_01.write('"\n')
    out_file_01.close()

    # Write predictions to csv file for analysis
    out_file_name_02 = 'sim_02.csv' 
    out_file_02 = open(out_file_name_02, 'a')

    for n in range(len(X_test)):
        out_file_02.write('"')
        out_file_02.write(str(y_pred[n]))
        out_file_02.write('","')
        out_file_02.write(str(X_test[n]))
        out_file_02.write('"\n')
    out_file_02.close()

# Classify hold back samples
t0 = time()

hb_file   = open('hb_01.csv') 
hb_reader = csv.reader(hb_file)
hb_samples = []
for new_line in hb_reader:
    hb_samples.append(new_line[0])
hb_file.close()

hb_test = tfidf.transform(hb_samples)
hb_pred = model.predict(hb_test)

print("done in %0.3fs" % (time() - t0))

# Write hold back prediction to file for analysis
out_hb_name = 'hb_02.csv' 
out_hb = open(out_hb_name, 'w')

for n in range(len(hb_samples)):
    out_hb.write('"')
    out_hb.write(str(hb_pred[n]))
    out_hb.write('","')
    out_hb.write(str(hb_samples[n]))
    out_hb.write('"\n')

out_hb.close()


