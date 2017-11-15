#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import re

import numpy as np

class EntityModel(object):

    def __init__(self,dict_path, dicts=['address','insurance_name']):
        assert dicts is not None and len(dicts) != 0, "empty dicts"

        self.dict_dir = dict_path
        self.dict_map = {}

        if os.path.exists(self.dict_dir):
            logging.info('loading entity dicts...')

            for dict_name in dicts:
                if os.path.exists(self.dict_dir+'/entity_model/'+dict_name+'.csv'):
                    logging.info('loading '+dict_name+' dictionary...')
                    self.dict_map[dict_name] = self.load_dict(self.dict_dir+'/entity_model/'+dict_name+'.csv')

    def rebuild_text(self,texts1, texts2):
        assert len(texts1) == len(texts2) or len(texts1) == 1, "invalid"

        pattern_texts1 = []
        pattern_texts2 = []

        if len(texts1) == 1:
            tmp_text1 = ''

            for dict_name in self.dict_map.keys():
                assert self.dict_map[dict_name] is not None, "address entity dictionary is missing..."

                text1_entitys = self.__get_entitys(texts1[0], self.dict_map[dict_name])
                tmp_text1 = self.__rebuild_text(texts1[0], text1_entitys, '@' + dict_name)

            tmp_text1, _ = self.number_pattern(tmp_text1, '')
            print tmp_text1

            for text2 in texts2:
                for dict_name in self.dict_map.keys():
                    assert self.dict_map[dict_name] is not None, "address entity dictionary is missing..."

                    text2_entitys = self.__get_entitys(text2, self.dict_map[dict_name])
                    text2 = self.__rebuild_text(text2, text2_entitys, '@' + dict_name)

                _, tmp_text2 = self.number_pattern('', text2)

                pattern_texts1.append(tmp_text1)
                pattern_texts2.append(tmp_text2)
        else:
            for text1, text2 in zip(texts1, texts2):

                for dict_name in self.dict_map.keys():
                    assert self.dict_map[dict_name] is not None, "address entity dictionary is missing..."

                    text1_entitys = self.__get_entitys(text1, self.dict_map[dict_name])
                    text2_entitys = self.__get_entitys(text2, self.dict_map[dict_name])

                    text1 = self.__rebuild_text(text1, text1_entitys, '@'+dict_name)
                    text2 = self.__rebuild_text(text2, text2_entitys, '@'+dict_name)

                tmp_text1, tmp_text2 = self.number_pattern(text1, text2)

                pattern_texts1.append(tmp_text1)
                pattern_texts2.append(tmp_text2)

        return pattern_texts1, pattern_texts2

    def __get_entitys(self, sentence, entitys):
        entitys_set = set()
        max_len = len(max(entitys, key=len))
        min_len = len(min(entitys, key=len))

        grams = set()
        for n in range(min_len,max_len):
            grams = set(zip(*[sentence[i:] for i in range(n)])) | grams

        for gram in grams:
            if gram in entitys:
                entitys_set.add(entitys)

        return entitys_set

    def number_pattern(self, text1, text2):
        number = re.compile(ur'\d+[wk百万亿]*')
        cn_number = re.compile(u'[一二三四五六七八九十百千万亿个]{1,}')

        tmp_text1 = ''
        tmp_text2 = ''

        t1_number_match = set(number.findall(text1))
        t1_cn_number_match = set(cn_number.findall(text1))
        t1_set = t1_number_match | t1_cn_number_match

        t2_number_match = set(number.findall(text2))
        t2_cn_number_match = set(cn_number.findall(text2))
        t2_set = t2_number_match | t2_cn_number_match

        tmp_text1 = tmp_text1 + self.__rebuild_text(text1, t1_set, '@number')
        tmp_text2 = tmp_text2 + self.__rebuild_text(text2, t2_set, '@number')

        return tmp_text1,tmp_text2

    def __rebuild_text(self,text, entitys, tag):
        for entity in entitys:
            text = text.replace(entity,tag)
        return text

    @classmethod
    def load_dict(self, filename):
        dict_set = set()
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                dict_set.add(unicode(line.strip(), 'utf-8'))
        return dict_set
