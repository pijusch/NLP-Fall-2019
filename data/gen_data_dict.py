import pickle
import gensim
import numpy as np
with open('entities_relations.pkl','rb') as f:
 a = pickle.load(f)

#w2v_model = dict()
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('~/GoogleNews-vectors-negative300.bin', binary=True)

ent = a[0]
rel = a[1]

ent_dic = dict()
rel_dic = dict()

for i in ent:
 if i in w2v_model:
  ent_dic[i] = w2v_model[i]
 elif i.lower() in w2v_model:
  ent_dic[i] = w2v_model[i.lower()]
 else:
  ent_dic[i] = np.zeros(300)

for i in rel:
 if i in w2v_model:
  rel_dic[i] = w2v_model[i]
 elif i.lower() in w2v_model:
  rel_dic[i] = w2v_model[i.lower()]
 else:
  rel_dic[i] = np.zeros(300)

with open('data_dicts.pkl','wb') as f:
 pickle.dump([ent_dic,rel_dic],f)
