arel = torch.FloatTensor(all_relations).cuda(0).squeeze(1)
arel, _ = model.rel_lstm(arel.squeeze(1))
arel = torch.mean(arel,1)

with open('syn_list_full.pkl','rb') as f:
  a = pickle.load(f)
with open('embeddings.pkl','rb') as f:
  b = pickle.load(f)

  def get_embeddings(x):
  tmp = []
  t = []
  n = 0
  for j in x['subject'].split():
    n+=1
    if n==3:
      break
    if j in b:
      t.append(b[j])
    else:
      t.append(np.zeros(300))
  tmp.append(t)
  t = []
  n = 0
  for j in x['object'].split():
    n+=1
    if n==3:
      break
    if j in b:
      t.append(b[j])
    else:
      t.append(np.zeros(300))
  tmp.append(t)
  t = []
  n = 0
  for j in x['predicate'].split():
    n+=1
    if n==3:
      break
    if j in b:
      t.append(b[j])
    else:
      t.append(np.zeros(300))
  tmp.append(t)
  for j in range(3):
    n = 3 - len(tmp[j])
    if n > 0:
      for k in range(n):
        tmp[j].append(np.zeros(300))

  return tmp

import random
mappings = []
aa = a.sample(n=100000)
for i in range(len(aa)):
  mappings.append(get_embeddings(aa.iloc[i]))
#get_embeddings(a.iloc[1])

subjects, objects, relations = torch.chunk(torch.FloatTensor(mappings).cuda(0), 3, dim=1)
sub = subjects.squeeze(1)
obj = objects.squeeze(1)
rel = relations.squeeze(1)
sub, _ = model.sub_lstm(sub.squeeze(1))
obj, _ = model.obj_lstm(obj.squeeze(1))
rel, _ = model.rel_lstm(rel.squeeze(1))
sub = torch.mean(sub,1)
obj = torch.mean(obj,1)
rel = torch.mean(rel,1)

arel = arel.cpu().detach().numpy()
sub = sub.cpu().detach().numpy()
obj = obj.cpu().detach().numpy()
rel = rel.cpu().detach().numpy()

hits1 = 0
hits5 = 0
hits10 = 0
hits100 = 0
for i in range(len(sub)):
  if i%1000==0:
    print(i)
  tmp = np.concatenate((arel,rel[i].reshape((1,100))))
  tmp = np.sum((sub[i]+tmp-obj[i])**2,axis=1)
  tmp.sort()
  ind = np.sum((sub[i]+rel[i]-obj[i])**2)
  ind = np.where(tmp==ind)[0][0]
  if ind<10:
    hits10+=1
  if ind<5:
    hits5+=1
  if ind<1:
    hits1+=1
  if ind<100:
    hits100+=1

print(hits1/len(sub))
print(hits5/len(sub))
print(hits10/len(sub))
print(hits100/len(sub))