#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import re

import numpy as np

class IntentModel(object):
    def __init__(self, dict_dir):
        self.dict_dir = dict_dir

        self.dict_path_address = self.dict_dir + '/fake_address_entity.csv'
        self.dict_path_insurance_name = self.dict_dir + '/fake_insurance_entity.csv'

        self.dict_path_pattern = self.dict_dir + '/fake_pattern.csv'

        self.dict_address = None
        self.dict_insurance_name = None

        # level two
        self.dict_pattern = None

        if os.path.exists(self.dict_dir):
            logging.info('loading intent dicts...')

            if os.path.exists(self.dict_path_address):
                logging.info('loading address entity dictionary...')
                self.dict_address = self.load_dict(self.dict_path_address)

            if os.path.exists(self.dict_path_insurance_name):
                logging.info('loading insurance_name entity dictionary...')
                self.dict_insurance_name = self.load_dict(self.dict_path_insurance_name)

            if os.path.exists(self.dict_path_pattern):
                logging.info('loading insurance_name entity dictionary...')
                self.dict_pattern = self.load_patten(self.dict_path_pattern)

    def similarity(self,texts1, texts2, dicts = ['address','insurance_name']):
        assert dicts is not None and len(dicts) != 0, "empty dicts"
        assert len(texts1) == len(texts2) or len(texts1) == 1, "invalid"

        features = []

        pattern_texts1 = []
        pattern_texts2 = []

        for text1, text2 in zip(texts1, texts2):

            feature_vec = []

            tmp_text1 = ''
            tmp_text2 = ''
            if 'address' in dicts:
                assert self.dict_address is not None, "address entity dictionary is missing..."
                text1_entitys = self.__get_entitys(text1, self.dict_address)
                text2_entitys = self.__get_entitys(text2, self.dict_address)
                feature_vec.append(len(text1_entitys & text2_entitys))

                tmp_text1 = tmp_text1 + self.__rebuild_text(text1, text1_entitys, '@address')
                tmp_text2 = tmp_text2 + self.__rebuild_text(text2, text2_entitys, '@address')


            if 'insurance_name' in dicts:
                assert self.dict_insurance_name is not None, "insurance_name entity dictionary is missing..."
                text1_entitys = self.__get_entitys(text1, self.dict_insurance_name)
                text2_entitys = self.__get_entitys(text2, self.dict_insurance_name)
                feature_vec.append(len(text1_entitys & text2_entitys))

                tmp_text1 = tmp_text1 + self.__rebuild_text(text1, text1_entitys, '@insurance')
                tmp_text2 = tmp_text2 + self.__rebuild_text(text2, text2_entitys, '@insurance')

            features.append(feature_vec)

            pattern_texts1.append(tmp_text1)
            pattern_texts2.append(tmp_text2)

        return np.array(features), pattern_texts1, pattern_texts2

    def __get_entitys(self, sentence, entitys):
        entitys_set = set()
        for entity in entitys:
            if entity in sentence:
                entitys_set.add(entity)
        return entitys_set

    def __rebuild_text(self,text, entitys, tag):
        for entity in entitys:
            text = text.replace(entity,tag)
        return text

    def __info_ex(self, sentence, pattens):
        pattern = ''
        for p in pattens:
            if p[1] in sentence:
                pattern = p[1]
                break
        return pattern

    def pattern_sim(self,texts1, texts2):
        features = []
        for text1_, text2_ in zip(texts1, texts2):
            text1_context = self.__info_ex(text1_, self.dict_pattern)
            text2_context = self.__info_ex(text2_, self.dict_pattern)
            if (text1_context == text2_context) and text1_context != '':
                features.append([1])
            else:
                features.append([0])
        return np.array(features)

    chs_arabic_map = {u'零': 0, u'一': 1, u'二': 2, u'三': 3, u'四': 4,
                      u'五': 5, u'六': 6, u'七': 7, u'八': 8, u'九': 9,
                      u'十': 10, u'百': 100, u'千': 10 ** 3, u'万': 10 ** 4,
                      u'〇': 0, u'壹': 1, u'贰': 2, u'叁': 3, u'肆': 4,
                      u'伍': 5, u'陆': 6, u'柒': 7, u'捌': 8, u'玖': 9,
                      u'拾': 10, u'佰': 100, u'仟': 10 ** 3, u'萬': 10 ** 4,
                      u'亿': 10 ** 8, u'億': 10 ** 8, u'幺': 1,
                      u'０': 0, u'１': 1, u'２': 2, u'３': 3, u'４': 4,
                      u'５': 5, u'６': 6, u'７': 7, u'８': 8, u'９': 9,
                      u'w': 10 ** 4, u'k': 10 ** 3, u'0': 0, u'1': 1, u'2': 2,
                      u'3': 3, u'4': 4, u'5': 5, u'6': 6, u'7': 7, u'8': 8, u'9': 9}

    def number_sim(self, text1, text2):
        number = re.compile(ur'\d+[wk百万亿]*')
        cn_number = re.compile(u'[一二三四五六七八九十百千万亿个]{1,}')

        result = []
        for text1_, text2_ in zip(text1, text2):
            features = []

            t1_set = set()
            t2_set = set()

            t1_number_match = set(number.findall(text1_))
            t1_cn_number_match = set(cn_number.findall(text1_))
            t1_set = t1_number_match | t1_cn_number_match
            if len(t1_set) != 0:
                t1_set = set([self.__convertChineseDigitsToArabic(t1) for t1 in t1_set])

            t2_number_match = set(number.findall(text2_))
            t2_cn_number_match = set(cn_number.findall(text2_))
            t2_set = t2_number_match | t2_cn_number_match
            if len(t2_set) != 0:
                t2_set = set([self.__convertChineseDigitsToArabic(t2) for t2 in t2_set])

            features.append(len(t1_set & t2_set))
            result.append(features)
        return np.array(result)

    def __convertChineseDigitsToArabic(self, chinese_digits, encoding="utf-8"):
        if isinstance(chinese_digits, str):
            chinese_digits = chinese_digits.decode(encoding)

        result = 0
        tmp = 0
        hnd_mln = 0
        for count in range(len(chinese_digits)):
            curr_char = chinese_digits[count]
            curr_digit = self.chs_arabic_map.get(curr_char, None)

            if curr_digit == 10 ** 8:
                result = result + tmp
                result = result * curr_digit
                hnd_mln = hnd_mln * 10 ** 8 + result
                result = 0
                tmp = 0
            elif curr_digit == 10 ** 4:
                result = result + tmp
                result = result * curr_digit
                tmp = 0
            elif curr_digit >= 10:
                tmp = 1 if tmp == 0 else tmp
                result = result + curr_digit * tmp
                tmp = 0
            elif curr_digit is not None:
                tmp = tmp * 10 + curr_digit
            else:
                return result
        result = result + tmp
        result = result + hnd_mln
        return unicode(str(result), 'utf-8')

    @classmethod
    def load_dict(self, filename):
        dict_set = set()
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                dict_set.add(unicode(line.strip(), 'utf-8'))
        return dict_set

    @classmethod
    def load_patten(self, filename):
        pingan_patten = []
        with open(filename) as f:
            for line in f:
                line = line.strip().decode('utf-8')
                line_list = line.split(' ')
                pingan_patten.append(line_list)
        return pingan_patten