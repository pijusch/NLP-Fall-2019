import numpy as np
import pickle
with open('avgembeddings.pkl','rb') as f:
  a = pickle.load(f)

s = a[0].cpu().detach().numpy()
p = a[2].cpu().detach().numpy()
o = a[1].cpu().detach().numpy()
rel = np.unique(p,axis=0)
ent = np.unique(np.concatenate((s,o)),axis=0)
hits = 0
for i in range(10000):
  if i%1000==0:
    print(i)
  e = np.sum((s[i]+rel-o[i])**2,axis=1)
  e.sort()
  p_ = np.sum((s[i]+p[i]-o[i])**2)
  indx = np.where(e==p_)[0]
  if len(indx)>0:
    indx = indx[0]
    if indx<10:
      hits+=1
  else:
    continue
print(hits/10000)

