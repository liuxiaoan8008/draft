#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import Levenshtein
import re

chs_arabic_map = {u'零': 0, u'一': 1, u'二': 2, u'三': 3, u'四': 4,
                  u'五': 5, u'六': 6, u'七': 7, u'八': 8, u'九': 9,
                  u'十': 10, u'百': 100, u'千': 10 ** 3, u'万': 10 ** 4,
                  u'〇': 0, u'壹': 1, u'贰': 2, u'叁': 3, u'肆': 4,
                  u'伍': 5, u'陆': 6, u'柒': 7, u'捌': 8, u'玖': 9,
                  u'拾': 10, u'佰': 100, u'仟': 10 ** 3, u'萬': 10 ** 4,
                  u'亿': 10 ** 8, u'億': 10 ** 8, u'幺': 1,
                  u'０': 0, u'１': 1, u'２': 2, u'３': 3, u'４': 4,
                  u'５': 5, u'６': 6, u'７': 7, u'８': 8, u'９': 9,
                  u'w':10 ** 4, u'k':10**3,u'0':0,u'1':1,u'2':2,
                  u'3': 3,u'4':4,u'5':5,u'6':6,u'7':7,u'8':8,u'9':9}

def edit_distance(texts1, texts2):
    if len(texts1) == 1:
        return np.array([[Levenshtein.distance(texts1[0], texts2[i])] for i in xrange(len(texts2))])
    else:
        return np.array([[Levenshtein.distance(texts1[i], texts2[i])] for i in xrange(len(texts2))])

def hamming_distance(texts1, texts2):
    if len(texts1) == 1:
        return np.array([[Levenshtein.hamming(texts1[0], texts2[i])] for i in xrange(len(texts2))])
    else:
        return np.array([[Levenshtein.hamming(texts1[i], texts2[i])] for i in xrange(len(texts2))])

def lcs(texts1, texts2):
    outputs = []
    for i in xrange(len(texts2)):
        mb = None
        if len(texts1) == 1:
            mb = Levenshtein.matching_blocks(Levenshtein.editops(texts1[0], texts2[i]), texts1[0], texts2[i])
        else:
            mb = Levenshtein.matching_blocks(Levenshtein.editops(texts1[i], texts2[i]), texts1[i], texts2[i])

        lcs_len = 0
        for x in mb:
           lcs_len += x[2]
        outputs.append([lcs_len])
    return np.array(outputs)

def ngram_similarity(texts1, texts2, n):

    assert len(texts1) == len(texts2) or len(texts1) == 1, "invalid"

    texts1_grams = set(zip(*[texts1[0][i:] for i in range(n)]))

    outputs = []
    for j in xrange(len(texts2)):
        if len(texts1) != 1 and j != 0:
            texts1_grams = set(zip(*[texts1[j][i:] for i in range(n)]))
        texts2_grams = (zip(*[texts2[j][i:] for i in range(n)]))

        join_grams = texts1_grams.intersection(texts2_grams)
        union_grams = texts1_grams.union(texts2_grams)

        similarity = 1.0
        if len(union_grams) > 0:
            similarity = 1.0 - (len(join_grams) / float(len(union_grams)))
        outputs.append([similarity])
    return np.array(outputs)

def load_patten(filename):
    pingan_patten = []
    with open(filename) as f:
        for line in f:
            line = line.strip().decode('utf-8')
            line_list = line.split(' ')
            pingan_patten.append(line_list)
    return pingan_patten

def info_ex(sentence,pattens):
    pattern = ''
    for p in pattens:
        if p[1] in sentence:
            pattern = p[1]
            break
    return pattern

def pattern_sim(text1,text2,pattens):
    features = []
    for text1_, text2_ in zip(text1,text2):
        text1_context = info_ex(text1_, pattens)
        text2_context = info_ex(text2_, pattens)
        if (text1_context == text2_context) and text1_context != '':
            features.append([1])
        else:
            features.append([0])
    return features

def load_dict(filename):
    dict_set = set()
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            dict_set.add(unicode(line.strip(),'utf-8'))
    return dict_set

def get_entitys(sentence,entitys):
    entitys_set = set()
    for entity in entitys:
        if entity in sentence:
            entitys_set.add(entity)
    return entitys_set

def entity_sim(text1, text2, entitys_map):
    result = []
    for text1_, text2_ in zip(text1, text2):
        features = []
        text1_entitys = get_entitys(text1_,entitys_map['address'])
        text2_entitys = get_entitys(text2_,entitys_map['address'])
        features.append(len(text1_entitys & text2_entitys))

        text1_entitys = get_entitys(text1_, entitys_map['insurance'])
        text2_entitys = get_entitys(text2_, entitys_map['insurance'])
        features.append(len(text1_entitys & text2_entitys))

        result.append(features)
    return np.array(result)

def get_number_sim(text1, text2):
    number = re.compile(ur'\d+[wk百万亿]*')
    cn_number = re.compile(u'[一二三四五六七八九十百千万亿个]{1,}')

    result = []
    for text1_,text2_ in zip(text1, text2):
        features = []

        t1_set = set()
        t2_set = set()

        t1_number_match = set(number.findall(text1_))
        t1_cn_number_match = set(cn_number.findall(text1_))
        t1_set = t1_number_match | t1_cn_number_match
        if len(t1_set) != 0:
            t1_set = set([convertChineseDigitsToArabic(t1) for t1 in t1_set])
        print t1_set

        t2_number_match = set(number.findall(text2_))
        t2_cn_number_match = set(cn_number.findall(text2_))
        t2_set = t2_number_match | t2_cn_number_match
        if len(t2_set) != 0:
            t2_set = set([convertChineseDigitsToArabic(t2) for t2 in t2_set])
        print t2_set
        print
        print t1_set & t2_set
        features.append(len(t1_set & t2_set))
        result.append(features)
    return result

def convertChineseDigitsToArabic(chinese_digits, encoding="utf-8"):
    if isinstance (chinese_digits, str):
        chinese_digits = chinese_digits.decode(encoding)

    result  = 0
    tmp     = 0
    hnd_mln = 0
    for count in range(len(chinese_digits)):
        curr_char  = chinese_digits[count]
        curr_digit = chs_arabic_map.get(curr_char, None)
        # meet 「亿」 or 「億」
        if curr_digit == 10 ** 8:
            result  = result + tmp
            result  = result * curr_digit
            # get result before 「亿」 and store it into hnd_mln
            # reset `result`
            hnd_mln = hnd_mln * 10 ** 8 + result
            result  = 0
            tmp     = 0
        # meet 「万」 or 「萬」
        elif curr_digit == 10 ** 4:
            result = result + tmp
            result = result * curr_digit
            tmp    = 0
        # meet 「十」, 「百」, 「千」 or their traditional version
        elif curr_digit >= 10:
            tmp    = 1 if tmp == 0 else tmp
            result = result + curr_digit * tmp
            tmp    = 0
        # meet single digit
        elif curr_digit is not None:
            tmp = tmp * 10 + curr_digit
        else:
            return result
    result = result + tmp
    result = result + hnd_mln
    return unicode(str(result),'utf-8')

print get_number_sim([u'能不能贷六十万'],[u'能不能贷60w'])
