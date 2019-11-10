import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from hits_eval import HitsEval


class WVEmbeddings:

    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        self.data = pd.read_csv('data/total.csv',sep='\t')
        num = len(self.data)
        self.data = self.data.iloc[int(0.8*num):int(0.9*num)]
        self.ent_dic = dict()
        self.rel_dic = dict()
        self.ent_dic_w2v = dict()
        self.rel_dic_w2v = dict()
        self.indices = np.ndarray((self.data.shape[0], self.data.shape[1]))
        self.set_embeddings()

    def set_embeddings(self):
        ent = list(set(list(self.data['0'])+list(self.data['1'])))
        rel = list(set(self.data['2']))

        for i in range(len(rel)):
            self.rel_dic[rel[i]] = i
            vec = np.zeros(300, dtype=float)
            count = 0
            for word in rel[i].split(' '):
                try:
                    vec += self.model[word.lower()]
                    count += 1
                except Exception as E:
                    # print(E)
                    pass
            self.rel_dic_w2v[i] = np.divide(vec, count)

        for i in range(len(ent)):
            self.ent_dic[ent[i]] = i
            vec = np.zeros(300, dtype=float)
            count = 0
            for word in ent[i].split(' '):
                try:
                    vec += self.model[word.lower()]
                    count += 1
                except Exception as E:
                    # print(E)
                    pass
            self.ent_dic_w2v[i] = np.divide(vec, count)

        for i in range(len(self.data)):
            self.indices[i][0] = self.ent_dic[self.data.iloc[i]['0']]
            self.indices[i][1] = self.ent_dic[self.data.iloc[i]['1']]
            self.indices[i][2] = self.rel_dic[self.data.iloc[i]['2']]

if __name__ == "__main__":
    wv  = WVEmbeddings()
    he = HitsEval(None, wv.indices, wv.ent_dic_w2v, wv.rel_dic_w2v)
    print(he.relations())