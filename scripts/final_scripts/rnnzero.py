import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from input_zero_shot import input_transe
import pickle

data = input_transe()

class TransE(nn.Module):
    def __init__(self, data, DIM_EMB=100, DIM_LSTM=300):
        super(TransE, self).__init__()
        self.dim = DIM_EMB
        self.num_ent = len(data.ent_dic)
        self.num_rel = len(data.rel_dic)
        self.linear_s = nn.Linear(3*DIM_EMB,3*DIM_EMB)
        self.linear_p = nn.Linear(3*DIM_EMB,3*DIM_EMB)
        self.linear_o = nn.Linear(3*DIM_EMB,3*DIM_EMB)
        self.linear_2_s = nn.Linear(3*DIM_EMB,DIM_EMB)
        self.linear_2_p = nn.Linear(3*DIM_EMB,DIM_EMB)
        self.sub_lstm = nn.RNN(DIM_LSTM,DIM_EMB,batch_first=True)
        self.obj_lstm = nn.RNN(DIM_LSTM,DIM_EMB,batch_first=True)
        self.rel_lstm = nn.RNN(DIM_LSTM,DIM_EMB,batch_first=True)
        self.ent_embedding = nn.Embedding(len(data.ent_dic)+1,DIM_EMB)
        self.rel_embedding = nn.Embedding(len(data.rel_dic)+1,DIM_EMB)
        self.embeddings = [self.ent_embedding, self.rel_embedding]
        self.initialize_embeddings()
        self.cuda(0)

    def normalize_embeddings(self):
        for e in self.embeddings:
            e.weight.data.renorm(p=2,dim=0,maxnorm=1)
    
    def initialize_embeddings(self):
        nn.init.xavier_uniform_(self.linear_s.weight)
        nn.init.xavier_uniform_(self.linear_p.weight)
        nn.init.xavier_uniform_(self.linear_o.weight)
        nn.init.xavier_uniform_(self.linear_2_s.weight)
        nn.init.xavier_uniform_(self.linear_2_p.weight)
        r = 6/np.sqrt(self.dim)
        for e in self.embeddings:
            e.weight.data.uniform_(-r, r)
        self.normalize_embeddings()

    def forward(self,subjects,objects,relations):
        sub = subjects.squeeze(1)
        obj = objects.squeeze(1)
        rel = relations.squeeze(1)
        #sub = self.ent_embedding(subjects).squeeze(1)
        #obj = self.ent_embedding(objects).squeeze(1)
        #rel = self.rel_embedding(relations).squeeze(1)
        sub, _ = self.sub_lstm(sub.squeeze(1))
        obj, _ = self.obj_lstm(obj.squeeze(1))
        rel, _ = self.rel_lstm(rel.squeeze(1))
        sub = torch.mean(sub,1)
        obj = torch.mean(obj,1)
        rel = torch.mean(rel,1)
        #sub = self.linear_2_s(self.linear_s(torch.flatten(sub,1)))
        #obj = self.linear_2_s(self.linear_s(torch.flatten(obj,1)))
        #rel = self.linear_2_p(self.linear_p(torch.flatten(rel,1)))
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
    tensor_spo = torch.FloatTensor(data.train).cuda(0)
    train_dataset = DataLoader(TensorDataset(tensor_spo, torch.zeros(tensor_spo.size(0))), batch_size=batch_n, shuffle=True, drop_last=True)
    for epoch in range(n_iter):
        total_loss = 0.0
        for batch_id, (spo, _) in enumerate(train_dataset):
            loss = transe_epoch(spo)
            #print(batch_id,loss)
            total_loss += loss
        print(f"loss on epoch {epoch} = {total_loss}")
    return

# def eval(data):
#     emb = []
#     c = 0
#     entities = model.ent_embedding(torch.LongTensor(range(model.num_ent+1)).cuda(0)).cpu().detach().numpy()
#     relations = model.rel_embedding(torch.LongTensor(range(model.num_rel+1)).cuda(0)).cpu().detach().numpy()
#     for i in np.array(data.test).astype(int):
#         t = []
#         t.append(entities[i[0]])
#         t.append(entities[i[1]])
#         t.append(relations[i[2]])
#         emb.append(t)
#         c+=1
#     he = HitsEval(emb,np.array(data.test).astype(int),entities,relations)
#     print(he.relations())

def hitsatk_transe(spo, k):

    total = 0.0
    s, o, p = torch.chunk(spo, 3, dim=1)
    #s_ = s.repeat(len(all_relations),1,1)
    s_ = s.repeat(1,len(all_relations),1,1).reshape((len(all_relations)*valid_batch,1,3,300))
    o_ = o.repeat(1,len(all_relations),1,1).reshape((len(all_relations)*valid_batch,1,3,300))
    e = torch.FloatTensor(all_relations).unsqueeze(1).cuda(0).repeat(valid_batch,1,1,1)
    #e = torch.LongTensor(all_relations).unsqueeze(1).repeat(valid_batch,1,1,1)
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

    tensor_spo = torch.FloatTensor(data).cuda(0)
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

def neg_gen():
    ns = torch.FloatTensor(np.array(random.sample(data.train,batch_n))[:,0]).cuda(0)
    no = torch.FloatTensor(np.array(random.sample(data.train,batch_n))[:,1]).cuda(0)
    return [ns,no]

if __name__ == "__main__":
    batch_n = 1024
    #data = input_transe()
    model = TransE(data)
    valid_batch = 64
    optimizer = optim.Adam(model.parameters())
    zero = torch.FloatTensor([0.0]).cuda(0)
    train_transe(data,20)
    torch.save(model,'./rnn.model')
    all_relations = data.rels
    model = torch.load('./rnn.model')
    run_transe_validation(data.test)
    # exit(0)
    # model = torch.load('./avg.model')
    #subjects, objects, relations = torch.chunk(torch.FloatTensor(data.valid).cuda(0), 3, dim=1)
    # sub = model.ent_embedding(subjects).squeeze(1)
    # obj = model.ent_embedding(objects).squeeze(1)
    # rel = model.rel_embedding(relations).squeeze(1)
    # #sub, _ = model.sub_lstm(sub.squeeze(1))
    # #obj, _ = model.obj_lstm(obj.squeeze(1))
    # #rel, _ = model.rel_lstm(rel.squeeze(1))
    # sub = torch.mean(sub,1)
    # obj = torch.mean(obj,1)
    # rel = torch.mean(rel,1)
    # print(torch.sum(torch.sum((sub+rel-obj)**2,-1)))
    # with open('valid_avg.pkl','wb') as f:
    #     pickle.dump([sub.cpu().detach().numpy(),obj.cpu().detach().numpy(),rel.cpu().detach().numpy()],f)
    #model.nerwork.cpu()
    #eval(data)
    #print(model.ent_embedding(torch.LongTensor([1])))
    # sub = subjects.squeeze(1)
    # obj = objects.squeeze(1)
    # rel = relations.squeeze(1)
    # sub = model.linear_2_s(model.linear_s(torch.flatten(sub,1)))
    # obj = model.linear_2_s(model.linear_s(torch.flatten(obj,1)))
    # rel = model.linear_2_p(model.linear_p(torch.flatten(rel,1)))
    # with open('zero.pkl','wb') as f:
    #   pickle.dump([sub, obj, rel],f)


