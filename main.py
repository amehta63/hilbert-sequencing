import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import CustomSequenceDataset

import torchsummary

import matplotlib.pyplot as plt

import os

from models import *
from utils import *
from train_model import *

BATCH_SIZE = 64
EPOCH_NUM = 30


if __name__ == "__main__":

    print("loading data...")
    
    training_data = CustomSequenceDataset()

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    print("started main at: " + getNowString())
    
    now = None

    name = ''
    
    if not now:
        now = getNowString()
        model=HilbertClassifier()

        #initialize and torchsummary
        tempindex, (tempsequence, tempmetadata) = next(enumerate(train_dataloader))
        model(hilbertCurve(tempsequence[0]).squeeze().unsqueeze(0).unsqueeze(0))
        torchsummary.summary(model, (1, 32, 32))

        print("training " + f"models/{model.__class__.__name__}_weights_{now}.pth")
        testModel, logs = train_model(model = model,
                        log_int=100, 
                        epochs=1, 
                        save_int=500, 
                        save_pth=f"models/{model.__class__.__name__}_weights_{now}.pth",
                        train_data=train_dataloader,
                        )
        f = open(f"models/{testModel.__class__.__name__}_structure_{now}.txt", "w")
        print(testModel, file=f)
        f.close()
        save_model(model=testModel, save_pth=f"models/{testModel.__class__.__name__}_{name}_full_{now}.pth")
    else:
        model_files = [x for x in os.listdir("models/") if now in x]
        full_models = [x for x in model_files if "full" in x]
        part_models = [x for x in model_files if "weights" in x]
        if full_models: 
            if len(full_models) > 1: raise Exception("model number not unique")
            print("loading " + "models/" + full_models[0])
            testModel = torch.load("models/"+full_models[0], weights_only=False)
            torchsummary.summary(testModel, (1, 32, 32))
        else:
            # TODO DELETE THESE
            testModel = HilbertClassifier()
            print("loading " + f"models/{testModel.__class__.__name__}_weights_{now}.pth")
            checkpoint = torch.load(f"models/{testModel.__class__.__name__}_weights_{now}.pth", weights_only=True)
            testModel.load_state_dict(checkpoint['model_state_dict'])

    testModel.eval()