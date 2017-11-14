import os
import logging

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
                text1_entitys = self.get_entitys(text1, self.dict_address)
                text2_entitys = self.get_entitys(text2, self.dict_address)
                feature_vec.append(len(text1_entitys & text2_entitys))

                tmp_text1 = tmp_text1 + self.rebuild_text(text1, text1_entitys, '@address')
                tmp_text2 = tmp_text2 + self.rebuild_text(text2, text2_entitys, '@address')


            if 'insurance_name' in dicts:
                assert self.dict_insurance_name is not None, "insurance_name entity dictionary is missing..."
                text1_entitys = self.get_entitys(text1, self.dict_insurance_name)
                text2_entitys = self.get_entitys(text2, self.dict_insurance_name)
                feature_vec.append(len(text1_entitys & text2_entitys))

                tmp_text1 = tmp_text1 + self.rebuild_text(text1, text1_entitys, '@insurance')
                tmp_text2 = tmp_text2 + self.rebuild_text(text2, text2_entitys, '@insurance')

            features.append(feature_vec)

            pattern_texts1.append(tmp_text1)
            pattern_texts2.append(tmp_text2)

        return np.array(features), pattern_texts1, pattern_texts2

    def get_entitys(self, sentence, entitys):
        entitys_set = set()
        for entity in entitys:
            if entity in sentence:
                entitys_set.add(entity)
        return entitys_set

    def rebuild_text(self,text, entitys, tag):
        for entity in entitys:
            text = text.replace(entity,tag)
        return text

    def info_ex(self, sentence, pattens):
        pattern = ''
        for p in pattens:
            if p[1] in sentence:
                pattern = p[1]
                break
        return pattern

    def pattern_sim(self,texts1, texts2):
        features = []
        for text1_, text2_ in zip(texts1, texts2):
            text1_context = self.info_ex(text1_, self.dict_pattern)
            text2_context = self.info_ex(text2_, self.dict_pattern)
            if (text1_context == text2_context) and text1_context != '':
                features.append([1])
            else:
                features.append([0])
        return np.array(features)

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