#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 22:07:46 2017

@author: janani
"""


from numpy import array
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN

def prepare_training_and_cross_validation_data(x, y, size=0.3, state=0):

    x_train, x_crossvalid, y_train, y_crossvalid = train_test_split(x, y, test_size=size, random_state=state)

    return (x_train, y_train, x_crossvalid, y_crossvalid)

def classify_using_knn(x_train, y_train, x_crossvalid, y_crossvalid, k=3):

    print("KNN")
    classifier = KNN(n_neighbors=k)
    classifier.fit(x_train, y_train)

    print ("Accuracy of training dataset: %f" % classifier.score(x_train, y_train))
    if y_crossvalid != None:
        print ("Accuracy of cross validation dataset: %f" % classifier.score(x_crossvalid, y_crossvalid))
    return classifier

def classify_using_bernoulli_naive_bayes(x_train, y_train, x_crossvalid, y_crossvalid):

    print ("Bernoulli Naive Bayes")
    classifier = BernoulliNB()
    classifier.fit(x_train, y_train)

    print ("Accuracy of training dataset: %f" % classifier.score(x_train, y_train))
    if y_crossvalid != None:
        print ("Accuracy of cross validation dataset: %f" % classifier.score(x_crossvalid, y_crossvalid))
    return classifier

def classify_using_naive_bayes(x_train, y_train, x_crossvalid, y_crossvalid):

    print ('Naive Bayes')
    classifier = MultinomialNB()
    classifier.fit(x_train, y_train)

    print ("Accuracy of training dataset: %f" % classifier.score(x_train, y_train))
    if y_crossvalid != None:
        print ("Accuracy of cross validation dataset: %f" % classifier.score(x_crossvalid, y_crossvalid))
    return classifier

def classify_using_random_forest(x_train, y_train, x_crossvalid, y_crossvalid):

    print ('Random Forest')
    classifier = RandomForest(n_estimators = 3000, max_features=2)
    classifier.fit(x_train, y_train)

    print ("Accuracy of training dataset: %f" % classifier.score(x_train, y_train))
    if y_crossvalid != None:
        print ("Accuracy of cross validation dataset: %f" % classifier.score(x_crossvalid, y_crossvalid))
    return classifier

def classify_using_logistic_regression(x_train, y_train, x_crossvalid, y_crossvalid):

    print ('Logistic Regression')
    classifier = LR(penalty='l2', C=.03)
    classifier.fit(x_train, y_train)

    print ("Accuracy of training dataset: %f" % classifier.score(x_train, y_train))
    if y_crossvalid != None:
        print ("Accuracy of cross validation dataset: %f" % classifier.score(x_crossvalid, y_crossvalid))
    return classifier

def get_probability_for_good_features(classifier, x):

    prob = array(classifier.predict_proba(x))
    return prob[:, 1]

if __name__ == '__main__':
    pass