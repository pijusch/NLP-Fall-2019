import pickle
import numpy as np
import pandas as pd
from hits_eval import HitsEval

class test():
    def __init__(self,b,a,total):
        total = total.iloc[:1000]
        ent = list(set(list(total['0'])+list(total['1'])))
        rel = list(set(total['2']))



        ent_dic = dict()
        rel_dic = dict()

        for i in range(len(rel)):
            rel_dic[i] = rel[i]
            rel_dic[rel[i]] = i

        for i in range(len(ent)):
            ent_dic[i] = ent[i]
            ent_dic[ent[i]] = i

        ent_emb = dict()
        rel_emb = dict()

        indices = np.ndarray((total.shape[0],total.shape[1]))
        for i in range(len(total)):
            indices[i][0] = ent_dic[total.iloc[i]['0']]
            indices[i][1] = ent_dic[total.iloc[i]['1']]
            indices[i][2] = rel_dic[total.iloc[i]['2']]
            ent_emb[indices[i][0]] = a[0][i,:]
            ent_emb[indices[i][1]] = a[1][i,:]
            rel_emb[indices[i][2]] = a[2][i,:]
        self.he = HitsEval(a,indices,ent_emb,rel_emb)
    
    def relations(self):
        return self.he.relations()
#with open('test_flatten.pkl','rb') as f:
#    a = pickle.load(f)

#a = np.array(a)
#c  = np.sum((a[0,:,:]+a[2,:,:]-a[1,:,:])**2,axis=1)
#print(sum((a[0][0]+a[2][0]-a[1][0])**2))
#print(sum(c))


#b = pd.read_csv('../data/total.csv',sep='\t')
#num = len(b)

#b = b.iloc[int(0.8*num):int(num*0.9)]
#b = b.iloc[:100000]


    
