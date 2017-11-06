#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import logging

from optparse import OptionParser

import jieba

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

if __name__ == '__main__':

    FORMAT = '[%(asctime)-15s] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = OptionParser()
    parser.add_option("-d", "--data", dest = "data", metavar = "FILE", help = "training data")
    parser.add_option("-o", "--output", dest = "output", metavar = "FILE", help = "output file")

    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    (options, args) = parser.parse_args()
    assert options.data is not None, "missing data"
    assert options.output is not None, "missing output"

    X = []
    with open(options.data, 'rb') as f:
        for line in f:
            X.append(u' '.join(jieba.lcut(line.strip())))

    clf = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1)
    count_clf = CountVectorizer()
    tfidf_clf = TfidfTransformer()

    count_res = count_clf.fit_transform(X)
    tfidf_res = tfidf_clf.fit_transform(count_res)
    clf.fit(tfidf_res)

    labels = clf.predict(tfidf_res)
    with open(options.output, 'w') as f:
        for i in xrange(len(X)):
            f.write(str(labels[i]) + '\t' + X[i].encode('utf8') + '\n')
        
