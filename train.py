from dataset import MyDataset
from model import MLPRegression
from visualize import plot_fig
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as op

data_tp=MyDataset("temps.csv")
loader=DataLoader(data_tp,batch_size=10,shuffle=True)

model_train=MLPRegression()
epochs=1000
criterion=nn.MSELoss()
optimizer=op.SGD(model_train.parameters(),lr=0.001)
loss=0
losses=[]
total=0

for epoch in range(epochs):
    for features,labels in loader:
        predict=model_train(features)
        label = labels.unsqueeze(1)
        loss=criterion(predict,label)
        total=total+loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss=total/len(loader)
    losses.append(avg_loss.item())
    total=0

eval_loader=DataLoader(data_tp,batch_size=10,shuffle=False)

all_predicts=[]
all_labels=[]

model_train.eval()

with torch.no_grad():
    for features,labels in eval_loader:
        predict=model_train(features)
        predict=predict.squeeze(1)
        all_predicts.extend(predict.numpy())
        all_labels.extend(labels.numpy())

plot_fig(losses,all_predicts,all_labels)