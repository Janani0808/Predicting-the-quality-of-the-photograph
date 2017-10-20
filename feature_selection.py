#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 22:15:33 2017

@author: janani
"""

from numpy import math

def generate_all_features_for_classifier(meta_data_train, y_train, meta_data_test):

    name_map_for_score, desc_map_for_score, caption_map_for_score, word_map_for_score = \
        generate_description_score_map(meta_data_train, y_train)
    geo_map_for_score, lat_map_for_score, lon_map_for_score = generate_location_score_map(
        meta_data_train, y_train)
    shape_map_for_score, size_map_for_score, width_map_for_score, height_map_for_score = \
            generate_description_score_map(meta_data_train, y_train)

    text_features_train = generate_description_features_for_classifier(meta_data_train,
        name_map_for_score, desc_map_for_score, caption_map_for_score, word_map_for_score)
    text_features_test = generate_description_features_for_classifier(meta_data_test,
        name_map_for_score, desc_map_for_score, caption_map_for_score, word_map_for_score)

    geo_features_train = generate_location_features_for_classifier(meta_data_train, geo_map_for_score,
        lat_map_for_score, lon_map_for_score)
    geo_features_test = generate_location_features_for_classifier(meta_data_test, geo_map_for_score,
        lat_map_for_score, lon_map_for_score)

    size_features_train = generate_overall_size_features_for_classifier(meta_data_train,
        shape_map_for_score, size_map_for_score, width_map_for_score, height_map_for_score)
    size_features_test = generate_overall_size_features_for_classifier(meta_data_test,
        shape_map_for_score, size_map_for_score, width_map_for_score, height_map_for_score)

    x_train = []
    for i in range(len(text_features_train)):
        x_train.append(text_features_train[i] + size_features_train[i] \
             + geo_features_train[i])

    x_test = []
    for i in range(len(text_features_test)):
        x_test.append(text_features_test[i] + size_features_test[i] \
            + geo_features_test[i])

    return (x_train, x_test)

def generate_location_features_for_classifier(meta_data, location_map_for_score, lat_map_for_score,
    lon_map_for_score):

    location_avg_score = get_map_average(location_map_for_score)
    lat_avg_score = get_map_average(lat_map_for_score)
    lon_avg_score = get_map_average(lon_map_for_score)

    location_score_features = []
    for line in meta_data:
        latitude = line[0]
        longitude = line[1]
        location = (latitude, longitude)

        location_score = location_avg_score
        if location in location_map_for_score:
            location_score = location_map_for_score[location]

        lat_score = lat_avg_score
        if latitude in lat_map_for_score:
            lat_score = lat_map_for_score[latitude]

        lon_score = lon_avg_score
        if longitude in lon_map_for_score:
            lon_score = lon_map_for_score[longitude]

        location_score_features.append([location_score, lat_score, lon_score])
    return location_score_features

def generate_location_score_map(meta_data, y):

    location_score_pairs = []
    lat_score_pairs = []
    lon_score_pairs = []
    for i in range(len(y)):
        latitude = meta_data[i][0]
        longitude = meta_data[i][1]
        geo = (latitude, longitude)

        location_score_pairs.append((geo, y[i]))
        lat_score_pairs.append((latitude, y[i]))
        lon_score_pairs.append((longitude, y[i]))

    geo_map_for_score = create_key_average_map(location_score_pairs)
    lat_map_for_score = create_key_average_map(lat_score_pairs)
    lon_map_for_score = create_key_average_map(lon_score_pairs)
    return (geo_map_for_score, lat_map_for_score, lon_map_for_score)

def generate_overall_size_features_for_classifier(meta_data, shape_map_for_score, size_map_for_score,
    width_map_for_score, height_map_for_score):

    avg_shape_score = get_map_average(shape_map_for_score)
    avg_size_score = get_map_average(size_map_for_score)
    avg_width_score = get_map_average(width_map_for_score)
    avg_height_score = get_map_average(height_map_for_score)

    size_score_features = []
    for line in meta_data:
        width = line[2]
        height = line[3]
        shape = (width, height)
        size = line[4]

        shape_score = avg_shape_score
        if shape in shape_map_for_score:
            shape_score = shape_map_for_score[shape]

        size_score = avg_size_score
        if size in size_map_for_score:
            size_score = size_map_for_score[size]

        width_score = avg_width_score
        if width in width_map_for_score:
            width_score = width_map_for_score[width]

        height_score = avg_height_score
        if height in height_map_for_score:
            height_score = height_map_for_score[height]

        size_score_features.append(
            [shape_score, size_score, width_score, height_score])
    return size_score_features

def generate_overall_size_score_map(meta_data, y):

    shape_score_pairs = []
    size_score_pairs = []
    width_score_pairs = []
    height_score_pairs = []
    for i in range(len(y)):
        width = meta_data[i][2]
        height = meta_data[i][3]
        shape = (width, height)
        size = meta_data[i][4]

        shape_score_pairs.append((shape, y[i]))
        size_score_pairs.append((size, y[i]))
        width_score_pairs.append((width, y[i]))
        height_score_pairs.append((height, y[i]))

    shape_map_for_score = create_key_average_map(shape_score_pairs)
    size_map_for_score = create_key_average_map(size_score_pairs)
    width_map_for_score = create_key_average_map(width_score_pairs)
    height_map_for_score = create_key_average_map(height_score_pairs)
    return (shape_map_for_score, size_map_for_score, width_map_for_score, height_map_for_score)

def generate_description_features_for_classifier(meta_data, name_map_for_score, desc_map_for_score,
    caption_map_for_score, word_map_for_score):

    avg_name_score = get_map_average(name_map_for_score)
    avg_desc_score = get_map_average(desc_map_for_score)
    avg_caption_score = get_map_average(caption_map_for_score)

    text_score_features = []
    for i in range(len(meta_data)):
        name = meta_data[i][5].split(' ')
        desc = meta_data[i][6].split(' ')
        caption = meta_data[i][7].split(' ')

        name_scores = []
        for s in name:
            if s in name_map_for_score:
                name_scores.append(name_map_for_score[s])
            elif s in word_map_for_score:
                name_scores.append(word_map_for_score[s])
            else:
                name_scores.append(avg_name_score)

        desc_scores = []
        for s in desc:
            if s in desc_map_for_score:
                desc_scores.append(desc_map_for_score[s])
            elif s in word_map_for_score:
                desc_scores.append(word_map_for_score[s])
            else:
                desc_scores.append(avg_desc_score)

        caption_scores = []
        for s in caption:
            if s in caption_map_for_score:
                caption_scores.append(caption_map_for_score[s])
            elif s in word_map_for_score:
                caption_scores.append(word_map_for_score[s])
            else:
                caption_scores.append(avg_caption_score)

        # Generates features.
        name_avg_score = float(sum(name_scores)) / len(name_scores)
        desc_avg_score = float(sum(desc_scores)) / len(desc_scores)
        caption_avg_score = float(sum(caption_scores)) / len(caption_scores)

        all_scores = name_scores + desc_scores + caption_scores
        total_avg_score = float(sum(all_scores)) / len(all_scores)

        name_std = std(name_scores, name_avg_score)
        desc_std = std(desc_scores, desc_avg_score)
        caption_std = std(caption_scores, caption_avg_score)
        total_std = std(all_scores, total_avg_score)

        name_len = 0
        if name[0] != '':
            name_len = len(name)
        desc_len = 0
        if desc[0] != '':
            desc_len = len(desc)
        caption_len = 0
        if caption[0] != '':
            caption_len = len(caption)

        text_score_features.append([name_avg_score, desc_avg_score,
            caption_avg_score, total_avg_score, name_len, desc_len,
            caption_len, name_std, desc_std, caption_std, total_std])
    return text_score_features

def generate_description_score_map(meta_data, y):

    name_y_pairs = []
    desc_y_pairs = []
    caption_y_pairs = []
    for i in range(len(y)):
        name = meta_data[i][5].split(' ')
        desc = meta_data[i][6].split(' ')
        caption = meta_data[i][7].split(' ')

        for s in name:
            name_y_pairs.append((s, y[i]))
        for s in desc:
            desc_y_pairs.append((s, y[i]))
        for s in caption:
            caption_y_pairs.append((s, y[i]))
    word_y_pairs = name_y_pairs + desc_y_pairs + caption_y_pairs

    name_map_for_score = create_key_average_map(name_y_pairs)
    desc_map_for_score = create_key_average_map(desc_y_pairs)
    caption_map_for_score = create_key_average_map(caption_y_pairs)
    word_map_for_score = create_key_average_map(word_y_pairs)
    return (name_map_for_score, desc_map_for_score, caption_map_for_score, word_map_for_score)

def std(iterable, avg):

    std = 0.0
    for n in iterable:
        std += (n - avg) ** 2
    return math.sqrt(std)

def get_map_average(key_value_map):

    average = 0.0
    for key in key_value_map.keys():
        average += key_value_map[key]
    return float(average) / len(key_value_map)

def create_key_average_map(key_value_pairs):

    key_average_map = {}
    for pair in key_value_pairs:
        k = pair[0]
        v = pair[1]
        if k not in key_average_map:
            key_average_map[k] = [v, 1]
        else:
            key_average_map[k][0] += v
            key_average_map[k][1] += 1

    for key in key_average_map.keys():
        key_average_map[key] = float(key_average_map[key][0]) / key_average_map[key][1]

    return key_average_map

if __name__ == '__main__':
    pass