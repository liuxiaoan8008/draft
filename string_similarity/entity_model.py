#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import re

import numpy as np

class EntityModel(object):

    DEFAULT_DICT_PATH = './data/dict'

    def __init__(self,dicts=['address','insurance_name']):
        assert dicts is not None and len(dicts) != 0, "empty dicts"

        self.dict_dir = self.DEFAULT_DICT_PATH
        self.dict_map = {}

        # level two
        # self.dict_path_pattern = self.dict_dir + '/fake_pattern.csv'
        # self.dict_pattern = None

        if os.path.exists(self.dict_dir):
            logging.info('loading intent dicts...')

            for dict_name in dicts:
                if os.path.exists(dict_name+'.csv'):
                    logging.info('loading '+dict_name+' dictionary...')
                    self.dict_map[dict_name] = self.load_dict(dict_name+'.csv')

    def similarity(self,texts1, texts2):
        assert len(texts1) == len(texts2) or len(texts1) == 1, "invalid"

        features = []

        for text1, text2 in zip(texts1, texts2):

            feature_vec = []

            for dict_name in self.dict_map.keys():
                assert self.dict_map[dict_name] is not None, "address entity dictionary is missing..."

                text1_entitys = self.__get_entitys(text1, self.dict_map[dict_name])
                text2_entitys = self.__get_entitys(text2, self.dict_map[dict_name])

                feature_vec.append(len((text1_entitys & text2_entitys))* 1.0 / len((text1_entitys | text2_entitys)))

            features.append(feature_vec)

        return np.array(features)

    def __get_entitys(self, sentence, entitys):
        entitys_set = set()
        for entity in entitys:
            if entity in sentence:
                entitys_set.add(entity)
        return entitys_set

    def number_sim(self, text1, text2):
        number = re.compile(ur'\d+[wk百万亿]*')
        cn_number = re.compile(u'[一二三四五六七八九十百千万亿个]{1,}')

        result = []
        for text1_, text2_ in zip(text1, text2):
            features = []

            t1_number_match = set(number.findall(text1_))
            t1_cn_number_match = set(cn_number.findall(text1_))
            t1_set = t1_number_match | t1_cn_number_match

            t2_number_match = set(number.findall(text2_))
            t2_cn_number_match = set(cn_number.findall(text2_))
            t2_set = t2_number_match | t2_cn_number_match

            if (t1_set | t2_set) >= 2:
                features.append(1)
            else:
                features.append(0)

            result.append(features)

        return np.array(result)

    @classmethod
    def load_dict(self, filename):
        dict_set = set()
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                dict_set.add(unicode(line.strip(), 'utf-8'))
        return dict_set
