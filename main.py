import torch
from torch.utils.data import DataLoader

from data_loader import CustomSequenceDataset

import torchsummary

from models import *
from utils import *
from train_model import *

import sys
import os
os.makedirs("data", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

BATCH_SIZE = 64
EPOCH_NUM = 10

class modelSelector(): # if you think i need getters and setters, bite me.
    def __init__(self, modeltype):
        self.modeltype = modeltype
        match modeltype:
            case '1D':
                self.model = Fluorescence1D.Fluorescence1D()
                self.datashape = (1, 450)
                self.datakey = 'sequences'
            case '1DESM':
                self.model = Fluorescence1D.Fluorescence1D(1280)
                self.datashape = (1280, 450)
                self.datakey = 'esm_sequences'
            case '2D':
                self.model = HilbertClassifier2D.HilbertClassifier2D()
                self.datashape = (1, 32, 32)
                self.datakey = 'hilbert_sequences'
            case '3D':
                self.model = HilbertClassifier3D.HilbertClassifier3D()
                self.datashape = (1, 1280, 32, 32)
                self.datakey = 'esm_hilbert_sequences'
            case _:
                raise ValueError("Model type not found.")


def run_model(modeltype = '1D', dataset = "data/dataloader_dict22104.pth", batch_size=BATCH_SIZE, epochs=EPOCH_NUM, now = None):

    model_selector = modelSelector(modeltype)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("loading data...")
    
    training_data = CustomSequenceDataset(saved_dataloader=dataset)
    training_data.prepTensorGetItem(model_selector.datakey)
    # training_data = CustomSequenceDataset(dimensions=3, datarange=1000, nAP_cond=10)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    print("started main at: " + getNowString())
    
    name = ''
    
    if not now:
        now = getNowString()
        model=model_selector.model.to(device)

        #initialize weights and torchsummary
        _, seqdict = next(enumerate(train_dataloader))
        tempsequence = seqdict[model_selector.datakey]
        tempsequence = tempsequence.to(device)
        model(tempsequence)
        torchsummary.summary(model, model_selector.datashape, depth=10)

        print("training " + f"checkpoints/{model.__class__.__name__}_weights_{now}.pth")
        testModel, logs = train_model(model = model,
                        log_int=100, 
                        epochs=epochs, 
                        save_int=500, 
                        save_pth=f"checkpoints/{model.__class__.__name__}_weights_{now}.pth",
                        train_data=train_dataloader,
                        dimensions=model_selector.modeltype,
                        device=device
                        )
        f = open(f"checkpoints/{testModel.__class__.__name__}_structure_{now}.txt", "w")
        print(testModel, file=f)
        f.close()
        save_model(model=testModel, save_pth=f"checkpoints/{testModel.__class__.__name__}_{name}_full_{now}.pth")
    else:
        model_files = [x for x in os.listdir("checkpoints/") if now in x]
        full_models = [x for x in model_files if "full" in x]
        part_models = [x for x in model_files if "weights" in x]
        if full_models: 
            if len(full_models) > 1: raise Exception("model number not unique")
            print("loading " + "checkpoints/" + full_models[0])
            testModel = torch.load("checkpoints/"+full_models[0], weights_only=False).to(device=device)
            torchsummary.summary(testModel, model_selector.datashape)

    testModel.eval()

if __name__ == "__main__":
    modeltype = '1D'
    if len(sys.argv) > 1:
        modeltype = sys.argv[1]
    if len(sys.argv) > 2:
        raise ValueError("Too many arguments.")
        
    print("running modeltype: " + modeltype)
    run_model(modeltype, now = None)
