import pandas as pd 
import numpy as np
import pickle
import os


class input_transe:
    def __init__(self):
        self.ent_dic = dict()
        self.rel_dic = dict()
        self.train = []
        self.test = []
        self.valid = []
        self.input()
    
    def input(self):
        if 'transEinput.pkl' in os.listdir():
            with open('transEinput.pkl','rb') as f:
                temp = pickle.load(f)
                self.train = temp[0]
                self.test = temp[1]
                self.valid = temp[2]
                self.rel_dic = temp[3]
                self.ent_dic = temp[4]
            return
        data = pd.read_csv('../data/total.csv',sep='\t')
        entities = list(set(list(data['0'])+list(data['1'])))
        relations = list(set(data['2']))
        
        for i in range(len(entities)):
            self.ent_dic[i] = entities[i]
            self.ent_dic[entities[i]] = i
        
        for i in range(len(relations)):
            self.rel_dic[i] = relations[i]
            self.rel_dic[relations[i]] = i

        out = np.ndarray((len(data),3))
        for i in range(len(data)):
            out[i][0] = self.ent_dic[data['0'].iloc[i]]
            out[i][1] = self.ent_dic[data['1'].iloc[i]]
            out[i][2] = self.rel_dic[data['2'].iloc[i]]
        self.train = out[0:int(n*0.8)]
        self.test = out[int(n*0.8):int(n*0.9)]
        self.valid = out[int(n*0.9):]
        with open('transEinput.pkl','wb') as f:
            pickle.dump([self.train,self.test,self.valid,self.rel_dic,self.ent_dic],f)
        


    def get_id(self,name,dic):
        return dic[name]
    
    def get_name(self,id,dic):
        return dic[id]
    
if __name__ == '__main__':
    n = input_transe()
    print(n.test[3:5])