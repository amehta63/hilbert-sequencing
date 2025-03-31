import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as functional
from torchvision.io import read_image

class CustomSequenceDataset(Dataset):
    """
    A dataset to convert gcamp protein well fluorescence readouts into dF, rise, and decay measures.
    Meant to work with pytorch dataloaders.

    Attributes:
        metadata (dict): metadata dict of tensors, each length of dataset from the following: id, variant, set, date, plate, well, mCherry, sequence, nAP, f0, dF, rise, decay.
        sequence (tensor): list of sequences, all zero_padded to 450 amino acids long.

    Methods:
        __len__(): returns number of sequences in dataset.
        __getitem__(idx): gets sequence and metadata at idx.
    """
    def __init__(self, data_file='data/train_gcamp3+6+8_90_well_mean_metrics_flattened.npz'):
        datadict = {}
        self.data = np.load(data_file)
        missingidx = np.isnan(self.data['dF'])
        missingidx = np.logical_or(missingidx, np.isnan(self.data['rise']))
        missingidx = np.logical_or(missingidx, np.isnan(self.data['decay']))
        self.missingidx = missingidx

        for i in self.data:
            datadict[i] = self.data[i][~self.missingidx]

        self.data = datadict

        self.metadata = {}
        self.seqlist = []
        for file in self.data.keys():
            if file == 'variant':
                self.metadata[file] = torch.FloatTensor(np.asarray(self.data[file], dtype=float)).squeeze().unsqueeze(1)
            elif file == 'sequence':
                for seq in self.data[file]:
                    self.seqlist.append(functional.pad(torch.FloatTensor([ord(x) for x in seq]), pad=(0, 450-len(seq))).squeeze().unsqueeze(1))
            else:
                self.metadata[file] = torch.FloatTensor(self.data[file]).squeeze().unsqueeze(1)
        self.seqlist = torch.hstack(self.seqlist).T

    def __len__(self):
        return len(self.seqlist)

    def __getitem__(self, idx):
        label = {k:v[idx] for k,v in self.metadata.items()}
        return self.seqlist[idx], label