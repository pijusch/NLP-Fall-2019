import pickle
import pandas as pd 

with open('elmo.pkl','rb') as f:
    ent = pickle.load(f)
rel = ent
#with open('rel.pkl','rb') as f:
#    rel = pickle.load(f)

tot = pd.read_csv('total.csv',sep ='\t')

en = list(tot['0'])+list(tot['1'])
re = list(tot['2'])
en = list(set(en))
re = list(set(re))

ent_dic = dict()
rel_dic = dict()
re_dic = dict()
en_dic = dict()

for i in range(len(en)):
    ent_dic[en[i]] = i
    en_dic[i] = ent[en[i]]
for i in range(len(re)):
    rel_dic[re[i]] = i
    re_dic[i] = rel[re[i]]

data = []
embed = []

for i in range(int(len(tot)/10)):
    if i%1000==0:
        print(i)
    t = []
    t.append(ent_dic[tot.iloc[i]['0']])
    t.append(ent_dic[tot.iloc[i]['1']])
    t.append(rel_dic[tot.iloc[i]['2']])
    data.append(t)
    t = []
    t.append(ent[tot.iloc[i]['0']])
    t.append(ent[tot.iloc[i]['1']])
    t.append(rel[tot.iloc[i]['2']])
    embed.append(t)

with open('elmo_data.pkl','wb') as f:
    pickle.dump([embed,data,en_dic,re_dic],f)
