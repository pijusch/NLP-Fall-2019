import sys
from input_transE import input_transe
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from input_transE import input_transe

class TransE(nn.Module):
    def __init__(self, data, DIM_EMB=100):
        super(TransE, self).__init__()
        self.dim = DIM_EMB
        self.num_ent = int(len(data.ent_dic)/2)
        self.num_rel = int(len(data.rel_dic)/2)
        self.ent_embedding = nn.Embedding(len(data.ent_dic),DIM_EMB)
        self.rel_embedding = nn.Embedding(len(data.rel_dic),DIM_EMB)
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
        sub = self.ent_embedding(subjects)
        obj = self.ent_embedding(objects)
        rel = self.rel_embedding(relations)
        score = torch.sum((sub+rel-obj)**2,-1)
        #score = torch.mul(score,score)
        #score = torch.sum(score,-1)
        return score.view(-1,1)

def transe_epoch(spo):
    s, o, p = torch.chunk(spo, 3, dim=1)
    no = torch.LongTensor(np.random.randint(0, model.num_ent, o.size(0))).view(-1,1).cuda(0)
    ns = torch.LongTensor(np.random.randint(0, model.num_ent, s.size(0))).view(-1,1).cuda(0)
    criterion = lambda pos, neg: torch.sum(torch.max(Variable(zero), 1 + pos - neg))
    
    optimizer.zero_grad()
    pos_score = model(Variable(s),Variable(o),Variable(p))
    neg_score = model(Variable(ns),Variable(o),Variable(p))
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
    prev = -10
    tensor_spo = torch.LongTensor(data.train).cuda(0)
    train_dataset = DataLoader(TensorDataset(tensor_spo, torch.zeros(tensor_spo.size(0))), batch_size=train_batch, shuffle=True, drop_last=True)
    for epoch in range(n_iter):
        total_loss = 0.0
        for batch_id, (spo, _) in enumerate(train_dataset):
            loss = transe_epoch(spo)
            #print(batch_id,loss)
            total_loss += loss
        print(f"loss on epoch {epoch} = {total_loss}")
        if epoch%10==0:
            hits = run_transe_validation(data.valid)
            if hits[2]>prev:
                prev = hits[2]
                torch.save(model,'./transE.model')

    return

def hitsatk_transe(spo, k):

    total = 0.0
    s, o, p = torch.chunk(spo, 3, dim=1)
    s = s.repeat(1, model.num_rel).view(-1, 1)
    o = o.repeat(1, model.num_rel).view(-1, 1)
    e = torch.LongTensor(np.arange(model.num_rel)).cuda(0).repeat(valid_batch).view(-1,1)
    #e = torch.LongTensor(np.arange(model.num_rel)).repeat(valid_batch).view(-1,1)
    output = model(Variable(s), Variable(o), Variable(e))
    output = output.view(-1, model.num_rel)

    hits = torch.nonzero((p == torch.topk(output, k, dim=-1, largest=False)[1].data).view(-1))
    if len(hits.size()) > 0:
        total += float(hits.size(0)) / p.size(0)
    return total

def run_transe_validation(data):

    tensor_spo = torch.LongTensor(data).cuda(0)
    valid_dataset = DataLoader(TensorDataset(tensor_spo, torch.zeros(tensor_spo.size(0))), batch_size=valid_batch, shuffle=True, drop_last=True)
    hits1 = []
    hits5 = []
    hits10 = []

    for batch_id, (spo, _) in enumerate(valid_dataset):

        print ("Validation batch ", batch_id)

        hits1 += [hitsatk_transe(spo, 1)]
        hits5 += [hitsatk_transe(spo, 5)]
        hits10 += [hitsatk_transe(spo, 10)]

    print( "Validation hits@1: %f" % (float(sum(hits1)) / len(hits1)))
    print( "Validation hits@10: %f" % (float(sum(hits5)) / len(hits5)))
    print( "Validation hits@100: %f" % (float(sum(hits10)) / len(hits10)))
    return [(float(sum(hits1)) / len(hits1)),(float(sum(hits5)) / len(hits5)),(float(sum(hits10)) / len(hits10))]


if __name__ == "__main__":
    data = input_transe()
    model = TransE(data)
    optimizer = optim.Adam(model.parameters())
    zero = torch.FloatTensor([0.0]).cuda(0)
    train_batch = 512
    valid_batch = 64
    train_transe(data,100)
    #torch.save(model,'./transE.model')
    model = torch.load('./transE.model')
    #run_transe_validation(data.valid)
    #model.nerwork.cpu()
    #print(model.ent_embedding(torch.LongTensor([1])))
