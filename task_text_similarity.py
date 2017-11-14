#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import csv
import logging

from optparse import OptionParser

from string_similarity.topic_model import TopicModel
from string_similarity.intent_model import IntentModel
from string_similarity.util import edit_distance, hamming_distance, ngram_similarity, lcs

import jieba
import numpy as np

from sklearn import metrics, preprocessing
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from xgboost import XGBClassifier

MODEL_FILENAME = 'text_similarity_xgboost_model.pkl'


def feature_extraction(x1, x2, mode, filepath, dictpath):
    global dict_map_set
    global dict_pattern

    seg_x1 = []
    for x in x1:
        seg_x1.append(jieba.lcut(x))

    seg_x2 = []
    for x in x2:
        seg_x2.append(jieba.lcut(x))

    if not hasattr(feature_extraction, "topic_model"):
        feature_extraction.topic_model = TopicModel(filepath)
    if mode == 'train':
        topic_model_training_data = []
        topic_model_training_data.extend(seg_x1)
        topic_model_training_data.extend(seg_x2)
        feature_extraction.topic_model.build(topic_model_training_data)

    if not hasattr(feature_extraction, "intent_model"):
        feature_extraction.intent_model = IntentModel(dictpath)

    extraction = feature_extraction.topic_model.similarity(seg_x1, seg_x2)
    extraction = np.concatenate((extraction, ngram_similarity(seg_x1, seg_x2, 2)), axis=1)
    extraction = np.concatenate((extraction, edit_distance(x1, x2)), axis=1)
    extraction = np.concatenate((extraction, lcs(x1, x2)), axis=1)

    number_sim = feature_extraction.intent_model.number_sim(x1, x2)
    extraction = np.concatenate((extraction, number_sim), axis=1)
    intent_sim,pattern_x1,pattern_x2 = feature_extraction.intent_model.similarity(x1, x2)
    extraction = np.concatenate((extraction, intent_sim), axis=1)

    # level two
    pattern_sim = feature_extraction.intent_model.pattern_sim(pattern_x1, pattern_x2)
    extraction = np.concatenate((extraction,pattern_sim),axis=1)
    extraction = np.concatenate((extraction, edit_distance(pattern_x1, pattern_x2)), axis=1)
    extraction = np.concatenate((extraction, lcs(pattern_x1, pattern_x2)), axis=1)

    return extraction


if __name__ == '__main__':

    FORMAT = '[%(asctime)-15s] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = OptionParser()
    parser.add_option("-d", "--data", dest="data", metavar="FILE", help="training/testing/candidate data")
    parser.add_option("-f", "--filepath", dest="filepath", metavar="FILE", help="model filepath")
    parser.add_option("-m", "--mode", dest="mode", help="interaction mode: train, test, try")
    parser.add_option("--dict", "--dictpath", dest="dictpath", metavar="FILE", help="dictionary filepath")

    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    (options, args) = parser.parse_args()
    assert options.mode is not None, "missing mode"
    if options.mode == "train":
        assert options.data is not None, "missing data"
        assert options.filepath is not None, "missing model filepath"
    elif options.mode == "test":
        assert options.data is not None, "missing data"
        assert options.filepath is not None, "missing model filepath"
    elif options.mode == "try":
        assert options.data is not None, "missing data"
        assert options.filepath is not None, "missing model filepath"
    else:
        assert False, "unknown mode"

    if not os.path.exists(options.filepath):
        os.makedirs(options.filepath)

    if options.mode == "train" or options.mode == "test":
        with open(options.data, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t')

            x1 = []
            x2 = []
            y = []
            for row in spamreader:
                assert len(row) == 3, "invalid row: " + str(row)
                x1.append(unicode(row[0], 'utf8'))
                x2.append(unicode(row[1], 'utf8'))
                y.append(int(row[2]))

            extraction = feature_extraction(x1, x2, options.mode, options.filepath, options.dictpath)
            if options.mode == "train":
                clf = Pipeline([
                    ('preprocess', preprocessing.StandardScaler()),
                    ('classifier', XGBClassifier(max_depth=7, min_child_weight=0.5, objective="binary:logistic"))
                ])
                clf.fit(extraction, y)
                joblib.dump(clf, options.filepath + '/' + MODEL_FILENAME)
            elif options.mode == "test":
                clf = joblib.load(options.filepath + '/' + MODEL_FILENAME)
                y_predict = clf.predict(extraction)
                logging.info(metrics.classification_report(y, y_predict))
                logging.info(metrics.accuracy_score(y, y_predict))
            else:
                assert False, "invalid"
    else:
        with open(options.data, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t')

            x = []
            for row in spamreader:
                assert len(row) > 1, "invalid row: " + row
                x.append(unicode(row[0], 'utf8'))

            while True:
                logging.info("please input a question: ")
                question = unicode(raw_input(), 'utf8')

                extraction = feature_extraction([question], x, options.mode, options.filepath, options.dictpath)
                clf = joblib.load(options.filepath + '/' + MODEL_FILENAME)
                y_predict = clf.predict_proba(extraction)

                x_cand = []
                for i in xrange(len(y_predict)):
                    x_cand.append((x[i], y_predict[i][1]))

                x_cand_sorted = sorted(x_cand, key=lambda k: 0 - k[1])
                for k in x_cand_sorted:
                    logging.info(k[0].encode('utf8'))
