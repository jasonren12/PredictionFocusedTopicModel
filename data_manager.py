import pickle

def save_dict(d, name):  
  file = open(name,'wb')
  pickle.dump(d, file)

def load_dict(name):
  file = open(name,'rb')
  d = pickle.load(file)
  return d

def load_Pang_Lee():
  return load_dict("datasets/Pang_Lee")
    