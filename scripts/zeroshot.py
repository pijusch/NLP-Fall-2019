from hits_eval import HitsEval
from synonyms import Syn
import pandas as pd

class ZeroShot():
    def __init__(self):
        self.vocab = Syn()
        self.total = pd.read_csv('../data/total.csv',sep='\t')
        return
    
    def relations(self):
        tmp = list(set(self.total['2']))
        l = []
        for i in tmp:
            if len(i.split())==1:
                l.append(i)
        l = self.total[self.total['2'].isin(l)]
        with open('predicate_one_word.txt','w') as f:
            for i in range(len(l)):
                lst = self.vocab.GetSyn(l.iloc[i]['2'])
                for j in range(len(lst)):
                    f.write(l.iloc[i][0]+'\t'+l.iloc[i][1]+'\t'+lst[j]+'\n')

if __name__ == "__main__":
    zs = ZeroShot()
    zs.relations()