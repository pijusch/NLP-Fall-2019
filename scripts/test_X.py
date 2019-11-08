import pickle
import numpy as np
import pandas as pd
from hits_eval import HitsEval

with open('test_X.pkl','rb') as f:
    a = pickle.load(f)

a = np.array(a)
c  = np.sum((a[0,:,:]+a[2,:,:]-a[1,:,:])**2,axis=1)
print(sum((a[0][0]+a[2][0]-a[1][0])**2))
print(sum(c))


exit(0)
b = pd.read_csv('../data/total.csv',sep='\t')
num = len(b)

b = b.iloc[int(0.8*num):int(0.9*num)]

ent = list(set(list(b['0'])+list(b['1'])))
rel = list(set(b['2']))

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

indices = np.ndarray((b.shape[0],b.shape[1]))
for i in range(len(b)):
    indices[i][0] = ent_dic[b.iloc[i]['0']]
    indices[i][1] = ent_dic[b.iloc[i]['1']]
    indices[i][2] = rel_dic[b.iloc[i]['2']]
    ent_emb[indices[i][0]] = a[0,i,:]
    ent_emb[indices[i][1]] = a[1,i,:]
    rel_emb[indices[i][2]] = a[2,i,:]
he = HitsEval(a,indices,ent_emb,rel_emb)
print(he.relations())
    