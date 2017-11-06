import os
import logging

import numpy as np
import gensim

class TopicModel(object):
    def __init__(self, model_dir):

        self.model_dir = model_dir

        self.model_path_dict = self.model_dir + '/model.dict'
        self.model_path_tfidf = self.model_dir + '/model.tfidf'
        self.model_path_lsi = self.model_dir + '/model.lsi'

        self.model_dict = None
        self.model_tfidf = None
        self.model_lsi = None

        if os.path.exists(self.model_dir):
            logging.info('loading topic model...')

            if os.path.exists(self.model_path_dict):
                logging.info('loading topic model(dictioanry)...')
                self.model_dict = gensim.corpora.Dictionary.load(self.model_path_dict)

            if os.path.exists(self.model_path_tfidf):
                logging.info('loading topic model(tfidf)...')
                self.model_tfidf = gensim.models.TfidfModel.load(self.model_path_tfidf)

            if os.path.exists(self.model_path_lsi):
                logging.info('loading topic model(lsi)...')
                self.model_lsi = gensim.models.LsiModel.load(self.model_path_lsi)

    def build(self, texts, modes=['tfidf', 'lsi']):
        assert modes is not None and len(modes) != 0, "empty modes"

        self.model_dict = gensim.corpora.Dictionary(texts)
        self.model_dict.save(self.model_path_dict)

        corpus = [self.model_dict.doc2bow(x) for x in texts]

        if 'tfidf' in modes:
            self.model_tfidf = gensim.models.TfidfModel(corpus)
            self.model_tfidf.save(self.model_path_tfidf)

        if 'lsi' in modes:
            assert self.model_tfidf is not None, "tfidf model is missing"

            corpus_tfidf = self.model_tfidf[corpus]
            self.model_lsi = gensim.models.LsiModel(corpus_tfidf, id2word=self.model_dict, num_topics = 50)
            self.model_lsi.save(self.model_path_lsi)

    def similarity(self, texts1, texts2, modes=['jaccard', 'tfidf', 'lsi']):
        assert modes is not None and len(modes) != 0, "empty modes"
        assert len(texts1) == len(texts2) or len(texts1) == 1, "invalid"

        assert self.model_dict is not None, "dict model is missing..."
        corpus1 = [self.model_dict.doc2bow(x) for x in texts1]
        corpus2 = [self.model_dict.doc2bow(x) for x in texts2]

        features = None

        tfidf1 = None
        tfidf2 = None
        if 'tfidf' in modes:
            assert self.model_tfidf is not None, "tfidf model is missing..."

            tfidf1 = self.model_tfidf[corpus1]
            tfidf2 = self.model_tfidf[corpus2]

        lsi1 = None
        lsi2 = None
        if 'lsi' in modes:
            assert self.model_lsi is not None, "lsi model is missing..."

            lsi1 = self.model_lsi[tfidf1]
            lsi2 = self.model_lsi[tfidf2]

        if len(texts1) == 1:
            features = self.concat(features, np.array([[
                           gensim.matutils.jaccard(corpus1[0], corpus2[i]) if 'jaccard' in modes else 0,
                           gensim.matutils.cossim(tfidf1[0], tfidf2[i]) if 'tfidf' in modes else 0,
                           gensim.matutils.cossim(lsi1[0], lsi2[i]) if 'lsi' in modes else 0
                       ] for i in xrange(len(corpus2))]))
        else:
            features = self.concat(features, np.array([[
                           gensim.matutils.jaccard(corpus1[i], corpus2[i]) if 'jaccard' in modes else 0,
                           gensim.matutils.cossim(tfidf1[i], tfidf2[i]) if 'tfidf' in modes else 0,
                           gensim.matutils.cossim(lsi1[i], lsi2[i]) if 'lsi' in modes else 0
                       ] for i in xrange(len(corpus2))]))
        return features

    @classmethod
    def concat(cls, arr1, arr2):
        if arr1 is None:
            return arr2
        else:
            return np.concatenate((arr1, arr2), axis=1)
