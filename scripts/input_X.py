import pandas as pd 
import numpy as np
import pickle
import random
import os

def get_labels(dic,x,num):
    re = []
    for i in x.split():
        re.append(dic[i])
    if len(re)>num:
        re = re[:num]
    elif len(re)<num:
        for i in range(len(re),num):
            re.append(len(dic))
    return re

class input_transe:
    def __init__(self):
        with open('../data/entities_relations.pkl','rb') as f:
            self.ent_rel = pickle.load(f)
        self.ent_dic = dict()
        self.rel_dic = dict()
        self.train = []
        self.test = []
        self.valid = []
        self.input()
    
    def input(self):
        if 'Xinput.pkl' in os.listdir():
            with open('Xinput.pkl','rb') as f:
                temp = pickle.load(f)
                self.train = temp[0]
                self.test = temp[1]
                self.valid = temp[2]
                self.rel_dic = temp[3]
                self.ent_dic = temp[4]
            return
        data = pd.read_csv('../data/total.csv',sep='\t')
        
        for i in range(len(self.ent_rel[0])):
            self.ent_dic[i] = self.ent_rel[0][i]
            self.ent_dic[self.ent_rel[0][i]] = i
        
        for i in range(len(self.ent_rel[1])):
            self.rel_dic[i] = self.ent_rel[1][i]
            self.rel_dic[self.ent_rel[1][i]] = i

        out = []
        for i in range(len(data)):
            tmp = []
            tmp.append(get_labels(self.ent_dic,data['0'].iloc[i],3))
            tmp.append(get_labels(self.ent_dic,data['1'].iloc[i],3))
            tmp.append(get_labels(self.rel_dic,data['2'].iloc[i],3))
            out.append(tmp)
        n = len(out)
        self.train = out[0:int(n*0.8)]
        self.test = out[int(n*0.8):int(n*0.9)]
        self.valid = out[int(n*0.9):]
        with open('Xinput.pkl','wb') as f:
            pickle.dump([self.train,self.test,self.valid,self.rel_dic,self.ent_dic],f)
        


    def get_id(self,name,dic):
        return dic[name]
    
    def get_name(self,id,dic):
        return dic[id]
    
if __name__ == '__main__':
    n = input_transe()
    print(n.train[0])
