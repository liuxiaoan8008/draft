import numpy as np
import Levenshtein

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




