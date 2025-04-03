import torch.nn as nn
from utils import *

# Calculate output of conv: [(input - kernel + 2*padding0)/stride1]+1
# Calculate output of conv': stride1*(input-1)+kernel-2*padding0 (literally the inverse lol)

class Fluorescence1D(nn.Module):
    def __init__(self, input_size=1):
        super().__init__()
        self.input_size = input_size

        self.convblock = nn.Sequential(
            nn.Conv1d(self.input_size, 120, 6, 1, 0),
            # nn.LazyConv1d(120, 6, 1, 0),
            nn.Dropout(0.2),
        )

        self.poolme = nn.Sequential(
            nn.Identity(),
            # nn.Linear(120, 240),
            # nn.ReLU(),
        )

        self.pool = self.globalMaxPool(dimension=2)

        self.lineardropout = nn.Sequential(
            nn.Linear(120, 100),
            nn.Dropout(0.2)
        )
        
        self.outputlayer = nn.Sequential(
            nn.Linear(100, 1),
            nn.Linear(1, 1),
            nn.Softplus()
        )

        self.dFreg = self.classifyingLinSeq(100, 1)
        self.risereg = self.classifyingLinSeq(100, 1)
        self.decayreg = self.classifyingLinSeq(100, 1)


        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.convblock(x)
        x = self.poolme(x)
        x = self.pool(x)
        x = self.lineardropout(x)
        dF = self.dFreg(x)
        rise = self.risereg(x)
        decay = self.decayreg(x)
        # x = self.relu(x)
        # x = self.relu(x)
        # x = self.sig(x)
        return dF, rise, decay

    def classifyingLinSeq(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Linear(output_size, output_size),
            nn.Softplus()
        )
    
    class globalMaxPool(nn.Module):
        def __init__(self, dimension=2):
            super().__init__()
            self.dimension = dimension
        def forward(self, x):
            return torch.max(x, dim=self.dimension)[0]