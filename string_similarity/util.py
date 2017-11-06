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
