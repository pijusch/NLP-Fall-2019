import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from input_transE import input_transe

class TransE(nn.Module):
    def __init__(self, X, Y, VOCAB_SIZE, DIM_EMB=10, NUM_CLASSES=2):
        super(TransE, self).__init__()
        (self.VOCAB_SIZE, self.DIM_EMB, self.NUM_CLASSES) = (VOCAB_SIZE, DIM_EMB, NUM_CLASSES)
        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.DIM_EMB)
        self.linear1 = nn.Linear(self.DIM_EMB,2)
        self.softmax = nn.Softmax(dim=0)

        #TODO: Initialize parameters.
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.linear1.weight)


    def forward(self, X, train=False):
        #TODO: Implement forward computation.
        return self.softmax(torch.sigmoid(self.linear1(torch.sigmoid(torch.mean(self.embedding(X),dim=0)))))

def Eval_TransE(X, Y, mlp):
    num_correct = 0
    for i in range(len(X)):
        logProbs = mlp.forward(X[i], train=False)
        pred = torch.argmax(logProbs)
        if pred == Y[i]:
            num_correct += 1
    print("Accuracy: %s" % (float(num_correct) / float(len(X))))

def Train_TransE(X, Y, vocab_size, n_iter):
    print("Start Training!")
    mlp = FFNN(X, Y, vocab_size)
    #TODO: initialize optimizer
    optimizer = optim.Adam(mlp.parameters(), lr=0.01)
    for epoch in range(n_iter):
        total_loss = 0.0
        for i in range(len(X)):
            x = X[i]
            y_onehot = torch.zeros(2)
            y_onehot[int(Y[i])] = 1
            mlp.zero_grad()
            probs = mlp.forward(x)
            #print(probs,y_onehot)
            loss = torch.neg(torch.log(probs)).dot(y_onehot)
            total_loss += loss
            
            loss.backward()
            optimizer.step()
            #TODO: compute gradients, do parameter update, compute loss.
        print(f"loss on epoch {epoch} = {total_loss}")
    return mlp

if __name__ == "__main__":
    train = IMDBdata("%s/train" % sys.argv[1])
    train.vocab.Lock()
    dev  = IMDBdata("%s/dev" % sys.argv[1], vocab=train.vocab)
    test  = IMDBdata("%s/test" % sys.argv[1], vocab=train.vocab)
    
    mlp = Train_FFNN(train.XwordList, (train.Y + 1.0) / 2.0, train.vocab.GetVocabSize(), int(sys.argv[2]))
    Eval_FFNN(dev.XwordList, (dev.Y + 1.0) / 2.0, mlp)
    Eval_FFNN(test.XwordList, (test.Y + 1.0) / 2.0, mlp)    
