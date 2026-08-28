import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing

class MyDataset(Dataset):
    def __init__(self,csv_path):
        super().__init__()
        self.data=pd.read_csv(csv_path)
        self.data=pd.get_dummies(self.data,columns=["week"],dtype=np.float32)
        self.features=self.data.drop(columns=["actual","friend"])
        self.features=preprocessing.StandardScaler().fit_transform(self.features)
        self.labels=self.data["actual"]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        features=self.features[index]
        labels=self.labels.iloc[index]
        features=torch.tensor(features,dtype=torch.float32)
        labels=torch.tensor(labels,dtype=torch.float32)
        return features,labels