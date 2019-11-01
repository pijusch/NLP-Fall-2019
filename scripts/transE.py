import sys
from input_transE import input_transe
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from input_transE import input_transe
from hits_eval import HitsEval

class TransE(nn.Module):
    def __init__(self, data, DIM_EMB=100):
        super(TransE, self).__init__()
        self.dim = DIM_EMB
        self.num_ent = len(data.ent_dic)
        self.num_rel = len(data.rel_dic)
        self.ent_embedding = nn.Embedding(len(data.ent_dic),DIM_EMB)
        self.rel_embedding = nn.Embedding(len(data.rel_dic),DIM_EMB)
        self.embeddings = [self.ent_embedding, self.rel_embedding]

    def normalize_embeddings(self):
        for e in self.embeddings:
            e.weight.data.renorm(p=2,dim=0,maxnorm=1)
    
    def intialize_embeddings(self):
        r = 6/np.sqrt(self.entity_dimensions)
        for e in self.embeddings:
            e.weight.data.uniform_(-r, r)
        self.normalize_embeddings()

    def forward(self,subjects,objects,relations):
        sub = self.ent_embedding(subjects)
        obj = self.ent_embedding(objects)
        rel = self.rel_embedding(relations)
        score = sub+rel-obj
        score = torch.mul(score,score)
        score = torch.sum(score,-1)
        return score.view(-1,1)

def transe_epoch(spo):
    s, o, p = torch.chunk(spo, 3, dim=1)
    no = torch.LongTensor(np.random.randint(0, model.num_ent, o.size(0))).view(-1,1).cuda(0)
    ns = torch.LongTensor(np.random.randint(0, model.num_ent, s.size(0))).view(-1,1).cuda(0)
    criterion = lambda pos, neg: torch.sum(torch.max(Variable(zero), 1.0 - pos + neg))
    
    optimizer.zero_grad()
    pos_score = model(Variable(s),Variable(o),Variable(p))
    neg_score = model(Variable(s),Variable(no),Variable(p))
    loss = criterion(pos_score,neg_score)
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
    tensor_spo = torch.LongTensor(data.index).cuda(x)
    train_dataset = DataLoader(TensorDataset(tensor_spo, torch.zeros(tensor_spo.size(0))), batch_size=1024, shuffle=True, drop_last=True)
    for epoch in range(n_iter):
        total_loss = 0.0
        for batch_id, (spo, _) in enumerate(train_dataset):
            loss = transe_epoch(spo)
            #print(batch_id,loss)
            total_loss += loss
        print(f"loss on epoch {epoch} = {total_loss}")
    return

def eval(data):
    emb = []
    c = 0
    entities = model.ent_embedding(torch.LongTensor(range(model.num_ent))).detach().numpy()
    relations = model.rel_embedding(torch.LongTensor(range(model.num_rel))).detach().numpy()
    for i in data.index[:10000].astype(int):
        if c%1000==0:
            print(c)
        t = []
        t.append(entities[i[0]])
        t.append(entities[i[1]])
        t.append(relations[i[2]])
        emb.append(t)
        c+=1
    he = HitsEval(emb,data.index[:10000].astype(int),entities,relations)
    print(he.relations())

if __name__ == "__main__":
    data = input_transe()
    model = TransE(data)
    optimizer = optim.Adam(model.parameters())
    zero = torch.FloatTensor([0.0]).cuda(0)
    train_transe(data,50)
    torch.save(model,'./transE.model')
    #model = torch.load('./transE.model')
    #eval(data)
    #print(model.ent_embedding(torch.LongTensor([1])))
