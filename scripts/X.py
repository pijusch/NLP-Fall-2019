import sys
import random
from input_transE import input_transe
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from input_X import input_transe
from hits_eval import HitsEval
import pickle

class TransE(nn.Module):
    def __init__(self, data, DIM_EMB=100, DIM_LSTM=100):
        super(TransE, self).__init__()
        self.dim = DIM_EMB
        self.num_ent = len(data.ent_dic)
        self.num_rel = len(data.rel_dic)
        self.sub_lstm = nn.LSTM(DIM_LSTM,DIM_EMB)
        self.obj_lstm = nn.LSTM(DIM_LSTM,DIM_EMB)
        self.rel_lstm = nn.LSTM(DIM_LSTM,DIM_EMB)
        self.ent_embedding = nn.Embedding(len(data.ent_dic)+1,DIM_EMB)
        self.rel_embedding = nn.Embedding(len(data.rel_dic)+1,DIM_EMB)
        self.embeddings = [self.ent_embedding, self.rel_embedding]
        self.initialize_embeddings()
        #self.cuda(0)

    def normalize_embeddings(self):
        for e in self.embeddings:
            e.weight.data.renorm(p=2,dim=0,maxnorm=1)
    
    def initialize_embeddings(self):
        r = 6/np.sqrt(self.dim)
        for e in self.embeddings:
            e.weight.data.uniform_(-r, r)
        self.normalize_embeddings()

    def forward(self,subjects,objects,relations):
        sub = self.ent_embedding(subjects)
        obj = self.ent_embedding(objects)
        rel = self.rel_embedding(relations)
        sub, _ = self.sub_lstm(sub.squeeze(1))
        obj, _ = self.obj_lstm(obj.squeeze(1))
        rel, _ = self.rel_lstm(rel.squeeze(1))
        sub = torch.mean(sub,1)
        obj = torch.mean(obj,1)
        rel = torch.mean(rel,1)
        score = torch.sum((sub+rel-obj)**2,-1)
        #score = torch.mul(score,score)
        #score = torch.sum(score,-1)
        return score.view(-1,1)

def transe_epoch(spo):
    s, o, p = torch.chunk(spo, 3, dim=1)
    no, ns = neg_gen()
    #no = torch.LongTensor(np.random.randint(0, model.num_ent, o.size(0))).view(-1,1).cuda(0)
    #ns = torch.LongTensor(np.random.randint(0, model.num_ent, s.size(0))).view(-1,1).cuda(0)
    criterion = lambda pos, neg: torch.sum(torch.max(Variable(zero), 1 + pos - neg))
    
    optimizer.zero_grad()
    pos_score = model(Variable(s),Variable(o),Variable(p))
    neg_score = model(Variable(ns),Variable(o),Variable(p))
    loss = criterion(pos_score,neg_score)
    #loss = torch.sum(pos_score)
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    pos_score = model(Variable(s),Variable(o),Variable(p))
    neg_score = model(Variable(s),Variable(no),Variable(p))
    loss = criterion(pos_score,neg_score)
    loss.backward()
    optimizer.step()

    return loss.item()

def train_transe(data, n_iter):
    print("Start Training!")
    tensor_spo = torch.LongTensor(data.test)
    train_dataset = DataLoader(TensorDataset(tensor_spo, torch.zeros(tensor_spo.size(0))), batch_size=batch_n, shuffle=True, drop_last=True)
    for epoch in range(n_iter):
        total_loss = 0.0
        for batch_id, (spo, _) in enumerate(train_dataset):
            loss = transe_epoch(spo)
            print(batch_id,loss)
            total_loss += loss
        print(f"loss on epoch {epoch} = {total_loss}")
    return

def eval(data):
    emb = []
    c = 0
    entities = model.ent_embedding(torch.LongTensor(range(model.num_ent))).detach().numpy()
    relations = model.rel_embedding(torch.LongTensor(range(model.num_rel))).detach().numpy()
    for i in data.train.astype(int):
        t = []
        t.append(entities[i[0]])
        t.append(entities[i[1]])
        t.append(relations[i[2]])
        emb.append(t)
        c+=1
    he = HitsEval(emb,data.train.astype(int),entities,relations)
    print(he.relations())

def neg_gen():
    ns = torch.LongTensor(np.array(random.sample(data.train,batch_n))[:,0])
    no = torch.LongTensor(np.array(random.sample(data.train,batch_n))[:,1])
    return [ns,no]

if __name__ == "__main__":
    batch_n = 4196
    data = input_transe()
    model = TransE(data)
    optimizer = optim.Adam(model.parameters())
    zero = torch.FloatTensor([0.0])
    #train_transe(data,10)
    #torch.save(model,'./transE.model')
    model = torch.load('./transE.model')
    subjects, objects, relations = torch.chunk(torch.LongTensor(data.valid), 3, dim=1)
    sub = model.ent_embedding(subjects)
    obj = model.ent_embedding(objects)
    rel = model.rel_embedding(relations)
    sub, _ = model.sub_lstm(sub.squeeze(1))
    obj, _ = model.obj_lstm(obj.squeeze(1))
    rel, _ = model.rel_lstm(rel.squeeze(1))
    sub = torch.mean(sub,1)
    obj = torch.mean(obj,1)
    rel = torch.mean(rel,1)
    with open('test_X.pkl','wb') as f:
          pickle.dump([sub.detach().numpy(),obj.detach().numpy(),rel.detach().numpy()],f)
    #model.nerwork.cpu()
    #eval(data)
    #print(model.ent_embedding(torch.LongTensor([1])))
