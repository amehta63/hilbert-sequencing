import torch.nn as nn
from utils import *

# Calculate output of conv: [(input - kernel + 2*padding0)/stride1]+1
# Calculate output of conv': stride1*(input-1)+kernel-2*padding0 (literally the inverse lol)

class HilbertClassifier3D(nn.Module):
    def __init__(self, input_size=1):
        super().__init__()
        self.input_size = input_size

        self.preClassifier = nn.Sequential(
            self.classifyingConvSeq(self.input_size, 32),     # ((32-3+2*1)/1)+1 = 32, 32->16 
            # self.classifyingConvSeq(32, 64),    # ((16-3+2*1)/1)+1 = 16, 16->8 
            #self.classifyingConvSeq(64, 128),   # ((8-3+2*1)/1)+1 = 8, 8->4
            #self.classifyingConvSeq(128, 256),  # ((4-3+2*1)/1)+1 = 4, 4->2
            #self.classifyingConvSeq(256, 512),  # ((2-3+2*1)/1)+1 = 2, 2->1
            nn.Flatten(),
            nn.LazyLinear(1000000),
        )

        self.dFreg = self.classifyingLinSeq(1)
        self.risereg = self.classifyingLinSeq(1)
        self.decayreg = self.classifyingLinSeq(1)

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.preClassifier(x)
        dF = self.dFreg(x)
        rise = self.risereg(x)
        decay = self.decayreg(x)
        # x = self.relu(x)
        # x = self.relu(x)
        # x = self.sig(x)
        return dF, rise, decay

    def classifyingConvSeq(self, in_channels, out_channels, kernel=3, padding=1):
        return nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size=kernel, stride=1, padding=padding
            ),  # no change to image size
            nn.BatchNorm3d(out_channels),  # no change to image size
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), padding=0), # image size * 0.5
        )

    def classifyingLinSeq(self, output_size):
        return nn.Sequential(
            nn.LazyLinear(100*output_size),
            # nn.Linear(2 * 2 * 256, 1024),
            nn.ReLU(),
            nn.Linear(100*output_size, 10*output_size),
            nn.ReLU(),
            nn.Linear(10*output_size, output_size),
            nn.ReLU(),
            #nn.LogSoftmax(dim=1),
        )
    