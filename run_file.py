#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 22:16:43 2017

@author: janani
"""

import time
import utilities
import classification
import feature_selection

def classification_of_all_features(training_file, num):
    """ Classifier using all features. """

    y, meta_data = utilities.read_from_training_file(training_file)
    y, meta_data = utilities.select_random_sample(y, meta_data, num)

    meta_data_train, y_train, meta_data_crossvalidation, y_crossvalidation = \
        classification.prepare_training_and_cross_validation_data(meta_data, y)

    x_train, x_crossvalidation = feature_selection.generate_all_features_for_classifier(meta_data_train,
        y_train, meta_data_crossvalidation)

    classifier = classification.classify_using_random_forest(x_train, y_train, x_crossvalidation, y_crossvalidation)
    print ("binomial deviance for training dataset")
    print (utilities.calculate_binomial_deviance_for_prediction(y_train,
        classification.get_probability_for_good_features(classifier, x_train)))
    print ("binomial deviance for cross validation dataset")
    print (utilities.calculate_binomial_deviance_for_prediction(y_crossvalidation, classification.get_probability_for_good_features(classifier, x_crossvalidation)))

    classifier = classification.classify_using_logistic_regression(x_train, y_train, x_crossvalidation, y_crossvalidation)
    print ("binomial deviance for training dataset")
    print (utilities.calculate_binomial_deviance_for_prediction(y_train,
        classification.get_probability_for_good_features(classifier, x_train)))
    print ("binomial deviance for cross validation dataset")
    print (utilities.calculate_binomial_deviance_for_prediction(y_crossvalidation, classification.get_probability_for_good_features(classifier, x_crossvalidation)))
    
    classifier = classification.classify_using_naive_bayes(x_train, y_train, x_crossvalidation, y_crossvalidation)
    print ("binomial deviance for training dataset")
    print (utilities.calculate_binomial_deviance_for_prediction(y_train,
        classification.get_probability_for_good_features(classifier, x_train)))
    print ("binomial deviance for cross validation dataset")
    print (utilities.calculate_binomial_deviance_for_prediction(y_crossvalidation, classification.get_probability_for_good_features(classifier, x_crossvalidation)))

    classifier = classification.classify_using_knn(x_train, y_train, x_crossvalidation, y_crossvalidation)
    print ("binomial deviance for training dataset")
    print (utilities.calculate_binomial_deviance_for_prediction(y_train,
        classification.get_probability_for_good_features(classifier, x_train)))
    print ("binomial deviance for cross validation dataset")
    print (utilities.calculate_binomial_deviance_for_prediction(y_crossvalidation, classification.get_probability_for_good_features(classifier, x_crossvalidation)))

    classifier = classification.classify_using_bernoulli_naive_bayes(x_train, y_train, x_crossvalidation, y_crossvalidation)
    print ("binomial deviance for training dataset")
    print (utilities.calculate_binomial_deviance_for_prediction(y_train,
        classification.get_probability_for_good_features(classifier, x_train)))
    print ("binomial deviance for cross validation dataset")
    print (utilities.calculate_binomial_deviance_for_prediction(y_crossvalidation, classification.get_probability_for_good_features(classifier, x_crossvalidation)))

def run_on_test_file(training_file, test_file, result_file):
    """ Running on the test file. """

    y, meta_data = utilities.read_from_training_file(training_file)
    ids, meta_data_test = utilities.read_from_test_file(test_file)

    x_train, x_test = feature_selection.generate_all_features_for_classifier(meta_data,
        y, meta_data_test)

    classifier = classification.classify_using_random_forest(x_train, y, None, None)
    probability = classification.get_probability_for_good_features(classifier, x_test)
    utilities.write_to_file(result_file, ids, probability)
    
    classifier = classification.classify_using_bernoulli_naive_bayes(x_train, y, None, None)
    probability = classification.get_probability_for_good_features(classifier, x_test)
    utilities.write_to_file(result_file, ids, probability)
    
    classifier = classification.classify_using_naive_bayes(x_train, y, None, None)
    probability = classification.get_probability_for_good_features(classifier, x_test)
    utilities.write_to_file(result_file, ids, probability)
    
    classifier = classification.classify_using_logistic_regression(x_train, y, None, None)
    probability = classification.get_probability_for_good_features(classifier, x_test)
    utilities.write_to_file(result_file, ids, probability)

if __name__ == '__main__':
    start_time = time.time()

    run_on_test_file('./data/training.csv', './data/test.csv', './data/result.csv')

    classification_of_all_features('./data/training.csv', 25000)

    print ((time.time() - start_time), 'seconds')
    
    