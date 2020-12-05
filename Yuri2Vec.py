import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from gensim.models import KeyedVectors as kv

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(200*2, 1000)
        self.fc2 = nn.Linear(1000,1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)

        x = torch.sigmoid(x)

        return x


        self.data = data
        self.labels = labels

wv = kv.load_word2vec_format("wiki.vec.pt", binary=True)

model = Model()
param = torch.load('weight.pth')
model.load_state_dict(param)

while True:
    name1 = input("AxBのAを入力してね:")
    name2 = input("AxBのBを入力してね:")

    vec1 = wv[name1[0]]
    for i in range(1, len(name1)):
        vec1 = vec1 + wv[name1[i]]
        vec1 = vec1.tolist()

    vec2 = wv[name2[0]]
    for i in range(1, len(name2)):
        vec2 = vec2 + wv[name2[i]]
        vec2 = vec2.tolist()

    vec = list(vec1)
    vec.extend(vec2)

    arr1 = np.array(vec)
    
    vec = list(vec2)
    vec.extend(vec1)

    arr2 = np.array(vec)

    data1 = torch.from_numpy(arr1.astype(np.float32)).clone()
    data2 = torch.from_numpy(arr2.astype(np.float32)).clone()

    out1 = model(data1)
    out2 = model(data2)

    yuri_vec = '→' if out1.item() > out2.item() else '←'

    print(name1 + " " + yuri_vec + " " + name2 + "\n\n")
