#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 22:17:47 2017

@author: janani
"""

import csv
from numpy import random
from numpy import math

def write_to_file(file_name, ids, p):

    print ("Writing to result file")
    open_file = open(file_name, 'w')
    writer_object = csv.writer(open_file)
    for i in range(len(p)):
        writer_object.writerow([ids[i], p[i]])
    open_file.close()

def read_from_test_file(file_name):
    #print 'Reading test file...'
    open_file = open(file_name)
    reader_object = csv.reader(open_file)
    next(reader_object)

    ids = []
    meta_data = []
    for column in reader_object:
        latitude_of_photograph = int(column[1])
        longitude_of_photograph = int(column[2])
        width_of_photograph = int(column[3])
        height_of_photograph = int(column[4])
        size_of_photograph = int(column[5])
        name_of_photograph = column[6]
        description_of_photograph = column[7]
        caption_of_photograph = column[8]

        ids.append(column[0])
        meta_data.append([latitude_of_photograph, longitude_of_photograph, width_of_photograph,
                          height_of_photograph, size_of_photograph, name_of_photograph,
                          description_of_photograph, caption_of_photograph])

    open_file.close()
    return (ids, meta_data)

def read_from_training_file(file_name):
    """ Reads training file and generates data. """

    #print 'Reading training file...'
    open_file = open(file_name)
    reader = csv.reader(open_file)
    next(reader)

    y = []
    meta_data = []
    for column in reader:
        latitude_of_photograph = int(column[1])
        longitude_of_photograph = int(column[2])
        width_of_photograph = int(column[3])
        height_of_photograph = int(column[4])
        size_of_photograph = int(column[5])
        name_of_photograph = column[6]
        description_of_photograph = column[7]
        caption_of_photograph = column[8]
        good_of_photograph = int(column[9])

        y.append(good_of_photograph)
        meta_data.append([latitude_of_photograph, longitude_of_photograph, width_of_photograph,
                          height_of_photograph, size_of_photograph, name_of_photograph,
                          description_of_photograph, caption_of_photograph])

    open_file.close()
    return (y, meta_data)

def select_random_sample(y, meta_data, random_number, randomly=True):
    """ Randomly samples num data from the whole data set. """

    if random_number == -1:
        random_number = len(y)
    y_sample = []
    meta_data_sample = []
    permuted_range = range(len(y))
    if randomly:
        permuted_range = random.permutation(len(y))
    permuted_range = permuted_range[0 : min(random_number, len(y))]
    for permuted_value in permuted_range:
        y_sample.append(y[permuted_value])
        meta_data_sample.append(meta_data[permuted_value])
    return (y_sample, meta_data_sample)

def calculate_binomial_deviance_for_prediction(y, predicted_value):
    """ Calculates the binomial deviance for the prediction. """

    binomial_deviance_for_prediction = 0.0
    for i in range(len(predicted_value)):
        if predicted_value[i] > .99:
            predicted_value[i] = .99
        elif predicted_value[i] < .1:
            predicted_value[i] = .1
        tmp = y[i] * math.log10(predicted_value[i])
        tmp += (1 - y[i]) * math.log10(1 - predicted_value[i])
        binomial_deviance_for_prediction -= tmp
    binomial_deviance_for_prediction /= float(len(predicted_value))
    return binomial_deviance_for_prediction

if __name__ == '__main__':
    pass