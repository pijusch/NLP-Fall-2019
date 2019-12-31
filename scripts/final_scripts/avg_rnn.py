import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from input_zero_shot2 import input_transe_z2
from input_X import input_transe
import pickle


class ConcatDataset(torch.utils.data.TensorDataset):
    def __init__(self, *dataset):
        super().__init__(*dataset)

    def __getitem__(self, index):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class TransE(nn.Module):
    def __init__(self, data, data_z, DIM_EMB=100, DIM_LSTM=300):
        super(TransE, self).__init__()
        self.dim = DIM_EMB
        self.num_ent = len(data.ent_dic)
        self.num_rel = len(data.rel_dic)
        self.linear_s = nn.Linear(3 * DIM_EMB, 3 * DIM_EMB)
        self.linear_p = nn.Linear(3 * DIM_EMB, 3 * DIM_EMB)
        self.linear_o = nn.Linear(3 * DIM_EMB, 3 * DIM_EMB)
        self.linear_2_s = nn.Linear(3 * DIM_EMB, DIM_EMB)
        self.linear_2_p = nn.Linear(3 * DIM_EMB, DIM_EMB)
        self.sub_lstm = nn.RNN(DIM_LSTM, DIM_EMB, batch_first=True)
        self.obj_lstm = nn.RNN(DIM_LSTM, DIM_EMB, batch_first=True)
        self.rel_lstm = nn.RNN(DIM_LSTM, DIM_EMB, batch_first=True)
        # self.ent_embedding_z = nn.Embedding(len(data_z.ent_dic)+1,DIM_EMB)
        # self.rel_embedding_z = nn.Embedding(len(data_z.rel_dic)+1,DIM_EMB)
        # self.embeddings_z = [self.ent_embedding_Z, self.rel_embedding_z]

        # self.num_ent_z = int(len(data.ent_dic) / 2)
        # self.num_rel_z = int(len(data.rel_dic) / 2)
        self.ent_embedding = nn.Embedding(len(data.ent_dic) + 1, DIM_EMB)
        self.rel_embedding = nn.Embedding(len(data.rel_dic) + 1, DIM_EMB)
        self.embeddings = [self.ent_embedding, self.rel_embedding]

        self.initialize_embeddings()
        self.cuda(0)

    def normalize_embeddings(self):
        for e in self.embeddings:
            e.weight.data.renorm(p=2, dim=0, maxnorm=1)
        # for e in self.embeddings_z:
        #     e.weight.data.renorm(p=2,dim=0,maxnorm=1)

    def initialize_embeddings(self):
        nn.init.xavier_uniform_(self.linear_s.weight)
        nn.init.xavier_uniform_(self.linear_p.weight)
        nn.init.xavier_uniform_(self.linear_o.weight)
        nn.init.xavier_uniform_(self.linear_2_s.weight)
        nn.init.xavier_uniform_(self.linear_2_p.weight)
        r = 6 / np.sqrt(self.dim)
        for e in self.embeddings:
            e.weight.data.uniform_(-r, r)
        # for e in self.embeddings_z:
        #     e.weight.data.uniform_(-r, r)
        self.normalize_embeddings()

    def forward(self, subjects_z, objects_z, relations_z, subjects, objects, relations):
        # print("forward")
        # print(subjects_z.shape)
        # print(subjects.shape)
        sub1 = self.ent_embedding(subjects)
        obj1 = self.ent_embedding(objects)
        rel1 = self.rel_embedding(relations)
        # print(sub1.shape)
        sub1 = sub1.squeeze(1)
        obj1 = obj1.squeeze(1)
        rel1 = rel1.squeeze(1)
        # print(sub1.shape)
        sub1 = torch.mean(sub1, 1)
        obj1 = torch.mean(obj1, 1)
        rel1 = torch.mean(rel1, 1)
        # print(sub1.shape)
        # sub1 = sub1.permute(1, 2, 0)[:, -1, :]
        # obj1 = obj1.permute(1, 2, 0)[:, -1, :]
        # rel1 = rel1.permute(1, 2, 0)[:, -1, :]
        sub = subjects_z.squeeze(1)
        obj = objects_z.squeeze(1)
        rel = relations_z.squeeze(1)
        # print(sub.shape)
        sub_z, _ = self.sub_lstm(sub.squeeze(1))
        obj_z, _ = self.obj_lstm(obj.squeeze(1))
        rel_z, _ = self.rel_lstm(rel.squeeze(1))
        # print(sub.shape)
        sub = torch.mean(sub, 1)
        obj = torch.mean(obj, 1)
        rel = torch.mean(rel, 1)
        # print(sub.shape)
        # print(sub1.shape)
        # sub_z = sub.t()
        # obj_z = obj.t()
        # rel = rel.t()
        # print(sub.shape)
        # print(sub1.shape)
        # sub = torch.flatten(sub,1)
        # obj = torch.flatten(obj,1)
        # rel = torch.flatten(rel,1)

        sub = torch.cat((sub, sub1), 1)
        # print(sub.shape)
        obj = torch.cat((obj, obj1), 1)
        # print(obj.shape)
        rel = torch.cat((rel, rel1), 1)
        # print(rel.shape)
        # exit(0)
        score = torch.sum((sub + rel - obj) ** 2, -1)
        # score = torch.mul(score,score)
        # score = torch.sum(score,-1)
        # print("score")
        # print(score.shape)
        # print("score view")
        # print(score.view(-1,1).shape)
        return score.view(-1, 1)


def transe_epoch(spo, spo_z):
    s_z, o_z, p_z = torch.chunk(spo_z, 3, dim=1)
    s, o, p = torch.chunk(spo, 3, dim=1)

    no, ns = neg_gen()
    ns_z = torch.FloatTensor(np.array(random.sample(data_z.train, batch_n))[:, 0]).cuda(0)
    no_z = torch.FloatTensor(np.array(random.sample(data_z.train, batch_n))[:, 1]).cuda(0)
    criterion = lambda pos, neg: torch.sum(torch.max(Variable(zero), 1 + pos - neg))
    # print("s_z")
    # print(s_z.shape)
    # print("ns_z")
    # print(ns_z.shape)
    optimizer.zero_grad()
    pos_score = model(Variable(s_z), Variable(o_z), Variable(p_z), Variable(s), Variable(o), Variable(p))
    neg_score = model(Variable(ns_z), Variable(o_z), Variable(p_z), Variable(ns), Variable(o), Variable(p))
    loss = criterion(pos_score, neg_score)
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    pos_score = model(Variable(s_z), Variable(o_z), Variable(p_z), Variable(s), Variable(o), Variable(p))
    neg_score = model(Variable(s_z), Variable(no_z), Variable(p_z), Variable(s), Variable(no), Variable(p))
    loss = criterion(pos_score, neg_score)
    loss.backward()
    optimizer.step()

    # optimizer.zero_grad()
    # pos_score = model(Variable(s), Variable(o), Variable(p))
    # neg_score = model(Variable(ns), Variable(o), Variable(p))
    # loss = criterion(pos_score, neg_score)
    # # loss = torch.sum(pos_score)
    # loss.backward()
    # optimizer.step()
    #
    # optimizer.zero_grad()
    # pos_score = model(Variable(s), Variable(o), Variable(p))
    # neg_score = model(Variable(s), Variable(no), Variable(p))
    # loss = criterion(pos_score, neg_score)
    # loss.backward()
    # optimizer.step()

    return loss.item()


def train_transe(data, data_z, n_iter):
    print("Start Training!")
    prev = -10
    tensor_spo = torch.LongTensor(data.train).cuda(0)
    tensor_spo_z = torch.FloatTensor(data_z.train).cuda(0)
    # # train_dataset = DataLoader(ConcatDataset(tensor_spo, torch.zeros(tensor_spo.size(0))),
    # #                                          batch_size=batch_n, shuffle=False, drop_last=True)
    # # train_dataset_z = DataLoader(ConcatDataset(tensor_spo_z, torch.zeros(tensor_spo_z.size(0))), batch_size=batch_n,
    # #                            shuffle=False, drop_last=True)
    # train_dataset = TensorDataset(tensor_spo, torch.zeros(tensor_spo.size(0)))
    # train_dataset_z = TensorDataset(tensor_spo_z, torch.zeros(tensor_spo_z.size(0)))
    # tensor_spo_z, ,torch.zeros(tensor_spo_z.size(0))
    # train_data = DataLoader(
    #          ConcatDataset(train_dataset, train_dataset_z),
    #          batch_size=batch_n, shuffle=True, drop_last=True)
    # train_dataset = DataLoader(TensorDataset(tensor_spo, torch.zeros(tensor_spo.size(0))), batch_size=batch_n, shuffle=True, drop_last=True)
    train_dataset = DataLoader(
        TensorDataset(tensor_spo, torch.zeros(tensor_spo.size(0)), tensor_spo_z, torch.zeros(tensor_spo_z.size(0))),
        batch_size=batch_n, shuffle=True, drop_last=True)

    for epoch in range(n_iter):
        total_loss = 0.0
        for batch_id, (spo, _, spo_z, __) in enumerate(train_dataset):
            # print()
            loss = transe_epoch(spo, spo_z)
            # print(batch_id,loss)
            total_loss += loss
        print(f"loss on epoch {epoch} = {total_loss}")
        if epoch % 10 == 0:
            hits = run_transe_validation(data.valid, data_z.valid)
            if hits[1] > prev:
                prev = hits[1]
                torch.save(model, './rnn.model')
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

def hitsatk_transe(spo, spo_z, k):
    # print("here")
    total = 0.0
    s, o, p = torch.chunk(spo, 3, dim=1)
    # s_ = s.repeat(len(all_relations),1,1)
    # print(s.shape)
    # s_ = s.repeat(1,len(all_relations),1,1)
    # print(s_.shape)
    # print("len")
    # print(len(all_relations)*valid_batch)
    # # print("v_batch")
    # # print(valid_batch)
    # s_=s_.reshape((len(all_relations)*valid_batch,1,3))
    # print(s_.shape)
    # exit()

    s_ = s.repeat(1, len(all_relations), 1).reshape((len(all_relations) * valid_batch, 1, 3))
    o_ = o.repeat(1, len(all_relations), 1).reshape((len(all_relations) * valid_batch, 1, 3))
    e = torch.LongTensor(all_relations).unsqueeze(1).cuda(0).repeat(valid_batch, 1, 1)
    # e = torch.LongTensor(all_relations).unsqueeze(1).repeat(valid_batch,1,1,1)
    # print(s_)
    # output = model(Variable(s_), Variable(o_), Variable(e))
    # output = output.view(-1, len(all_relations))
    # print(torch.topk(output, k, dim=-1, largest=False)[0].data)
    # print(output[0].sort())
    # print(p)

    s_z, o_z, p_z = torch.chunk(spo_z, 3, dim=1)
    # s_ = s.repeat(len(all_relations),1,1)
    sz_ = s_z.repeat(1, len(all_relations_z), 1, 1).reshape((len(all_relations_z) * valid_batch, 1, 3, 300))
    oz_ = o_z.repeat(1, len(all_relations_z), 1, 1).reshape((len(all_relations_z) * valid_batch, 1, 3, 300))
    ez = torch.FloatTensor(all_relations_z).unsqueeze(1).cuda(0).repeat(valid_batch, 1, 1, 1)
    # e = torch.LongTensor(all_relations_z).unsqueeze(1).repeat(valid_batch,1,1,1)
    # print(s_)
    output = model(Variable(sz_), Variable(oz_), Variable(ez), Variable(s_), Variable(o_), Variable(e))
    output = output.view(-1, len(all_relations_z))

    p = model(Variable(s_z), Variable(o_z), Variable(p_z), Variable(s), Variable(o), Variable(p))

    hits = torch.nonzero((p == torch.topk(output, k, dim=-1, largest=False)[0].data).view(-1))
    if len(hits.size()) > 0:
        total += float(hits.size(0)) / p.size(0)
    return total


def run_transe_validation(data, data_z):
    tensor_spo = torch.LongTensor(data).cuda(0)
    tensor_spo_z = torch.FloatTensor(data_z).cuda(0)
    # valid_dataset = DataLoader(TensorDataset(tensor_spo, torch.zeros(tensor_spo.size(0))), batch_size=valid_batch, shuffle=False, drop_last=True)
    valid_dataset = DataLoader(
        TensorDataset(tensor_spo, torch.zeros(tensor_spo.size(0)), tensor_spo_z, torch.zeros(tensor_spo_z.size(0))),
        batch_size=valid_batch, shuffle=True, drop_last=True)
    hits1 = []
    hits5 = []
    hits10 = []

    for batch_id, (spo, _, spo_z, __) in enumerate(valid_dataset):
        # print("spo")
        # print(spo.shape)
        # print("spo_z")
        # print(spo_z.shape)
        # print ("Validation batch ", batch_id)

        hits1 += [hitsatk_transe(spo, spo_z, 1)]
        hits5 += [hitsatk_transe(spo, spo_z, 5)]
        hits10 += [hitsatk_transe(spo, spo_z, 10)]

    print("Validation hits@1: %f" % (float(sum(hits1)) / len(hits1)))
    print("Validation hits@5 %f" % (float(sum(hits5)) / len(hits5)))
    print("Validation hits@10: %f" % (float(sum(hits10)) / len(hits10)))
    return [(float(sum(hits1)) / len(hits1)), (float(sum(hits5)) / len(hits5)), (float(sum(hits10)) / len(hits10))]


def neg_gen():
    ns = torch.LongTensor(np.array(random.sample(data.train, batch_n))[:, 0]).cuda(0)
    no = torch.LongTensor(np.array(random.sample(data.train, batch_n))[:, 1]).cuda(0)
    return [ns, no]


if __name__ == "__main__":
    batch_n = 1024
    data = input_transe()
    data_z = input_transe_z()
    model = TransE(data, data_z)
    valid_batch = 64
    all_relations = data.rels
    all_relations_z = data_z.rels
    optimizer = optim.Adam(model.parameters())
    zero = torch.FloatTensor([0.0]).cuda(0)
    train_transe(data, data_z, 100)
    torch.save(model, './rnn_emd.model')
    # model = torch.load('./rnn.model')
    # run_transe_validation(data.test)
    # exit(0)
    # model = torch.load('./avg.model')
    # subjects, objects, relations = torch.chunk(torch.FloatTensor(data.valid).cuda(0), 3, dim=1)
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
    # model.nerwork.cpu()
    # eval(data)
    # print(model.ent_embedding(torch.LongTensor([1])))
    # sub = subjects.squeeze(1)
    # obj = objects.squeeze(1)
    # rel = relations.squeeze(1)
    # sub = model.linear_2_s(model.linear_s(torch.flatten(sub,1)))
    # obj = model.linear_2_s(model.linear_s(torch.flatten(obj,1)))
    # rel = model.linear_2_p(model.linear_p(torch.flatten(rel,1)))
    # with open('zero.pkl','wb') as f:
    #   pickle.dump([sub, obj, rel],f)