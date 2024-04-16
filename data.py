import json
import pickle

ec_lst = pickle.load(open('/scratch0/zx22/zijie/esm/data/ec_type.pkl', 'rb'))

with open("/scratch0/zx22/zijie/esm/data/train_cut.json","r") as fr:
    data = json.load(fr)

print(len(list(set(ec_lst))))
print(len(ec_lst))
