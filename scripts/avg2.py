import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from input_X import input_transe
import pickle

class TransE(nn.Module):
    def __init__(self, data, DIM_EMB=100, DIM_LSTM=100):
        super(TransE, self).__init__()
        self.dim = DIM_EMB
        self.num_ent = int(len(data.ent_dic)/2)
        self.num_rel = int(len(data.rel_dic)/2)
        self.linear = nn.Linear(3*self.dim,self.dim)
        #self.sub_lstm = nn.LSTM(DIM_LSTM,DIM_EMB)
        #self.obj_lstm = nn.LSTM(DIM_LSTM,DIM_EMB)
        #self.rel_lstm = nn.LSTM(DIM_LSTM,DIM_EMB)
        self.ent_embedding = nn.Embedding(len(data.ent_dic)+1,DIM_EMB)
        self.rel_embedding = nn.Embedding(len(data.rel_dic)+1,DIM_EMB)
        self.embeddings = [self.ent_embedding, self.rel_embedding]
        self.initialize_embeddings()
        self.cuda(0)

    def normalize_embeddings(self):
        for e in self.embeddings:
            e.weight.data.renorm(p=2,dim=0,maxnorm=1)
    
    def initialize_embeddings(self):
        r = 6/np.sqrt(self.dim)
        for e in self.embeddings:
            e.weight.data.uniform_(-r, r)
        self.normalize_embeddings()

    def forward(self,subjects,objects,relations):
        sub = self.ent_embedding(subjects).squeeze(1)
        obj = self.ent_embedding(objects).squeeze(1)
        rel = self.rel_embedding(relations).squeeze(1)
        #sub, _ = self.sub_lstm(sub.squeeze(1))
        #obj, _ = self.obj_lstm(obj.squeeze(1))
        #rel, _ = self.rel_lstm(rel.squeeze(1))
        sub = self.linear(torch.flatten(sub,1))
        obj = self.linear(torch.flatten(obj,1))
        rel = self.linear(torch.flatten(rel,1))
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
    prev = -10
    tensor_spo = torch.LongTensor(data.train).cuda(0)
    train_dataset = DataLoader(TensorDataset(tensor_spo, torch.zeros(tensor_spo.size(0))), batch_size=batch_n, shuffle=True, drop_last=True)
    for epoch in range(n_iter):
        total_loss = 0.0
        for batch_id, (spo, _) in enumerate(train_dataset):
            loss = transe_epoch(spo)
            #print(batch_id,loss)
            total_loss += loss
        print(f"loss on epoch {epoch} = {total_loss}")
        if epoch%10==0:
            hits = run_transe_validation(data.valid)
            if hits[1]>prev:
                prev = hits[1]
                torch.save(model,'./avg2.model')
    return

# def eval(data):
#     subjects, objects, relations = torch.chunk(torch.LongTensor(np.concatenate((data.train,data.test,data.valid))), 3, dim=1)
#     sub = model.ent_embedding(subjects).squeeze(1)
#     obj = model.ent_embedding(objects).squeeze(1)
#     rel = model.rel_embedding(relations).squeeze(1)
#     sub = torch.mean(sub,1)
#     obj = torch.mean(obj,1)
#     rel = torch.mean(rel,1)
#     num = len(total)
#     out = np.array([sub,obj,rel])
#     evaluation = test(total.iloc[int(0.8*num):int(0.9*num)],out,total )
#     print(evaluation.relations())

def neg_gen():
    ns = torch.LongTensor(np.array(random.sample(data.train,batch_n))[:,0]).cuda(0)
    no = torch.LongTensor(np.array(random.sample(data.train,batch_n))[:,1]).cuda(0)
    return [ns,no]

def hitsatk_transe(spo, k):

    total = 0.0
    s, o, p = torch.chunk(spo, 3, dim=1)
    #s_ = s.repeat(len(all_relations),1,1)
    s_ = s.repeat(1,len(all_relations),1).reshape((len(all_relations)*valid_batch,1,3))
    o_ = o.repeat(1,len(all_relations),1).reshape((len(all_relations)*valid_batch,1,3))
    e = torch.LongTensor(all_relations).unsqueeze(1).cuda(0).repeat(valid_batch,1,1)
    #e = torch.LongTensor(all_relations).unsqueeze(1).repeat(valid_batch,1,1)
    # print(s_)
    output = model(Variable(s_), Variable(o_), Variable(e))
    output = output.view(-1, len(all_relations))
    
    p = model(Variable(s), Variable(o), Variable(p))
    # print(torch.topk(output, k, dim=-1, largest=False)[0].data)
    # print(output[0].sort())
    # print(p)

    hits = torch.nonzero((p == torch.topk(output, k, dim=-1, largest=False)[0].data).view(-1))
    if len(hits.size()) > 0:
        total += float(hits.size(0)) / p.size(0)
    return total

def run_transe_validation(data):

    tensor_spo = torch.LongTensor(data).cuda(0)
    valid_dataset = DataLoader(TensorDataset(tensor_spo, torch.zeros(tensor_spo.size(0))), batch_size=valid_batch, shuffle=False, drop_last=True)
    hits1 = []
    hits10 = []
    hits100 = []

    for batch_id, (spo, _) in enumerate(valid_dataset):

        print ("Validation batch ", batch_id)

        hits1 += [hitsatk_transe(spo, 1)]
        hits10 += [hitsatk_transe(spo, 10)]
        hits100 += [hitsatk_transe(spo, 100)]

    print( "Validation hits@1: %f" % (float(sum(hits1)) / len(hits1)))
    print( "Validation hits@10: %f" % (float(sum(hits10)) / len(hits10)))
    print( "Validation hits@100: %f" % (float(sum(hits100)) / len(hits100)))
    return [(float(sum(hits1)) / len(hits1)),(float(sum(hits10)) / len(hits10)),(float(sum(hits100)) / len(hits100))]

if __name__ == "__main__":
    batch_n = 1024
    valid_batch = 1024
    data = input_transe()
    model = TransE(data)
    optimizer = optim.Adam(model.parameters())
    all_relations = data.rels
    zero = torch.FloatTensor([0.0]).cuda(0)
    train_transe(data,100)
    #torch.save(model,'./avg.model')
    #exit(0)
    #model = torch.load('./avg.model')
    run_transe_validation(data.test)
    # subjects, objects, relations = torch.chunk(torch.LongTensor(np.concatenate((data.train, data.test, data.valid))).cuda(0), 3, dim=1)
    # sub = model.ent_embedding(subjects).squeeze(1)
    # obj = model.ent_embedding(objects).squeeze(1)
    # rel = model.rel_embedding(relations).squeeze(1)
    # sub = torch.mean(sub,1)
    # obj = torch.mean(obj,1)
    # rel = torch.mean(rel,1)
    # with open('avgembeddings.pkl','wb') as f:
    #   pickle.dump([sub,obj,rel],f)
