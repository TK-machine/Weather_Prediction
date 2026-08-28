import torch.nn as nn

class MLPRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,16),
            nn.ReLU(),

            nn.Linear(16,1)
        )
    def forward(self,x):
        return self.model(x)