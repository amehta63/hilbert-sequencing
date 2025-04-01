import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import CustomSequenceDataset

import torchsummary

import os

from models import *
from utils import *
from train_model import *

BATCH_SIZE = 64
EPOCH_NUM = 10


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("loading data...")
    
    training_data = CustomSequenceDataset(saved_dataloader="data/dataloader_dict.pth")
    # training_data = CustomSequenceDataset()
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

    print("started main at: " + getNowString())
    
    now = None

    name = ''
    
    if not now:
        now = getNowString()
        model=HilbertClassifier().to(device)

        #initialize weights and torchsummary
        _, (_, tempsequence, _) = next(enumerate(train_dataloader))
        tempsequence = tempsequence.to(device)
        model(tempsequence)
        torchsummary.summary(model, (1, 32, 32))

        print("training " + f"models/{model.__class__.__name__}_weights_{now}.pth")
        testModel, logs = train_model(model = model,
                        log_int=100, 
                        epochs=EPOCH_NUM, 
                        save_int=500, 
                        save_pth=f"models/{model.__class__.__name__}_weights_{now}.pth",
                        train_data=train_dataloader,
                        device=device
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
            testModel = torch.load("models/"+full_models[0], weights_only=False).to(device=device)
            torchsummary.summary(testModel, (1, 32, 32))
        else:
            # TODO DELETE THESE
            testModel = HilbertClassifier().to(device=device)
            print("loading " + f"models/{testModel.__class__.__name__}_weights_{now}.pth")
            checkpoint = torch.load(f"models/{testModel.__class__.__name__}_weights_{now}.pth", weights_only=True)
            testModel.load_state_dict(checkpoint['model_state_dict'])

    testModel.eval()