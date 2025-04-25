import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import CustomSequenceDataset

import torchsummary

import os

from Fluorescence1D import Fluorescence1D
from utils import *
from train_model import *

BATCH_SIZE = 64
EPOCH_NUM = 10


def run_1desm_model(dataset = "data/dataloader_dict22104.pth", batch_size=BATCH_SIZE, epochs=EPOCH_NUM):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("loading data...")
    
    training_data = CustomSequenceDataset(saved_dataloader=dataset)
    # training_data = CustomSequenceDataset(dimensions=3, datarange=1000, nAP_cond=10)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    print("started main at: " + getNowString())
    
    now = None

    name = ''
    
    if not now:
        now = getNowString()
        model=Fluorescence1D(1280).to(device)

        #initialize weights and torchsummary
        _, seqdict = next(enumerate(train_dataloader))
        tempsequence = seqdict['esm_sequences']
        tempsequence = tempsequence.to(device)
        model(tempsequence.permute(0,2,1))
        torchsummary.summary(model, (1280, 450), depth=10)

        print("training " + f"models/{model.__class__.__name__}_weights_{now}.pth")
        testModel, logs = train_model(model = model,
                        log_int=100, 
                        epochs=epochs, 
                        save_int=500, 
                        save_pth=f"models/{model.__class__.__name__}_weights_{now}.pth",
                        train_data=train_dataloader,
                        dimensions='1DESM',
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
            testModel = Fluorescence1D().to(device=device)
            print("loading " + f"models/{testModel.__class__.__name__}_weights_{now}.pth")
            checkpoint = torch.load(f"models/{testModel.__class__.__name__}_weights_{now}.pth", weights_only=True)
            testModel.load_state_dict(checkpoint['model_state_dict'])

    testModel.eval()

if __name__ == "__main__":
    run_1desm_model()