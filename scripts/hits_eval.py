import pandas as pd 
import numpy as np
import pickle
import torch

class HitsEval():
    def __init__(self,embeddings,data,ent_dic,rel_dic):
        self.ent_block = []
        self.rel_block = []
        self.ent_dict = ent_dic
        self.rel_dict = rel_dic
        self.ent_num = len(self.ent_dict)
        self.rel_num = len(self.rel_dict)
        for i in range(self.ent_num):
            self.ent_block.append(self.ent_dict[i])
        for i in range(self.rel_num):
            self.rel_block.append(self.rel_dict[i])
        self.ent_block = torch.stack(self.ent_block)
        self.rel_block = torch.stack(self.rel_block)
        self.embed = embeddings
        self.data = data
        return

    def heads(self):
        hits1 = 0
        hits5 = 0
        hits10 = 0
        imr = 0
        for i in range(len(self.data)):
            if i%1000==0:
                print(i)
            tmp = (self.ent_block + self.rel_dict[self.data[i][2]]- self.ent_dict[self.data[i][1]])**2
            tmp = np.sum(tmp,axis=1)
            idx  = np.where(np.argsort(tmp)==self.data[i][0])[0][0]
            imr += (1/(idx+1))
            if idx == 0:
                hits1+=1
            if idx<5:
                hits5+=1
            if idx<10:
                hits10+=1
        
        return [hits1/len(self.data),hits5/len(self.data),hits10/len(self.data),imr/len(self.data)]
        return

    def tails(self):
        hits1 = 0
        hits5 = 0
        hits10 = 0
        imr = 0
        for i in range(len(self.data)):
            if i%1000==0:
                print(i)
            tmp = (self.ent_dict[self.data[i][0]]+self.rel_dict[self.data[i][2]]-self.ent_block)**2
            tmp = np.sum(tmp,axis=1)
            idx  = np.where(np.argsort(tmp)==self.data[i][1])[0][0]
            imr += (1/(idx+1))
            if idx == 0:
                hits1+=1
            if idx<5:
                hits5+=1
            if idx<10:
                hits10+=1
        
        return [hits1/len(self.data),hits5/len(self.data),hits10/len(self.data),imr/len(self.data)]

    def relations(self):
        hits1 = 0
        hits5 = 0
        hits10 = 0
        imr = 0
        for i in range(len(self.data)):
            if i%1000==0:
                print(i)
            tmp = (self.ent_dict[self.data[i][0]] + self.rel_block - self.ent_dict[self.data[i][1]])**2
            tmp = torch.sum(tmp,dim=0).detach().numpy()
            #tmp = np.sum(tmp,axis=1)
            idx  = np.where(np.argsort(tmp)==self.data[i][2])
            print(idx)
            idx = idx[0][0]
            imr += (1/(idx+1))
            if idx == 0:
                hits1+=1
            if idx<5:
                hits5+=1
            if idx<10:
                hits10+=1
        
        return [hits1/len(self.data),hits5/len(self.data),hits10/len(self.data),imr/len(self.data)]
        return

if __name__ == "__main__":
    with open('kg_data.pkl','rb') as f:
        tmp = pickle.load(f)
    he = HitsEval(tmp[0][:],tmp[1][:],tmp[2],tmp[3])
    print(he.tails())
