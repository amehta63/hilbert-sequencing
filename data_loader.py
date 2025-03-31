import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as functional
from utils import *


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
    def __init__(self, data_file='data/train_gcamp3+6+8_90_well_mean_metrics_flattened.npz', sequence_length=450, saved_dataloader=None):
        
        try:
            dataloader_dict = torch.load(saved_dataloader, weights_only=True)
        except Exception as e:
            if saved_dataloader is not None:
                raise e
            print("No existing dataset passed, loading full dataset. This will take up to 15 minutes.")
            self.sequence_length = sequence_length
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
                        self.seqlist.append(functional.pad(torch.FloatTensor([ord(x) for x in seq]), pad=(0, self.sequence_length-len(seq))).squeeze().unsqueeze(1))
                else:
                    self.metadata[file] = torch.FloatTensor(self.data[file]).squeeze().unsqueeze(1)
            self.seqlist = torch.hstack(self.seqlist).T

            self.p = (int(np.ceil(np.sqrt(self.sequence_length)))-1).bit_length()

            self.hilbert_seqlist = []
            for seq in self.seqlist:
                sequence = functional.pad(seq, (0, 2**(2*self.p)-self.sequence_length))
                twoDseq = hilbertCurve(sequence).squeeze()
                twoDseq = twoDseq.reshape(-1, 2**self.p, 2**self.p)
                self.hilbert_seqlist.append(twoDseq)
            self.hilbert_seqlist = torch.cat(self.hilbert_seqlist, dim=0).unsqueeze(1)

            dataloader_dict = {}
            dataloader_dict['sequence_length'] = self.sequence_length
            # dataloader_dict['data'] = self.data
            dataloader_dict['missingidx'] = torch.from_numpy(self.missingidx)
            dataloader_dict['metadata'] = self.metadata
            dataloader_dict['seqlist'] = self.seqlist
            dataloader_dict['p'] = self.p
            dataloader_dict['hilbert_seqlist'] = self.hilbert_seqlist
            torch.save(dataloader_dict, 'data/dataloader_dict.pth')
        else:
            self.sequence_length = dataloader_dict['sequence_length']
            #self.data = dataloader_dict['data']
            self.missingidx = dataloader_dict['missingidx']
            self.metadata = dataloader_dict['metadata']
            self.seqlist = dataloader_dict['seqlist']
            self.p = dataloader_dict['p']
            self.hilbert_seqlist = dataloader_dict['hilbert_seqlist']

    def __len__(self):
        return len(self.seqlist)

    def __getitem__(self, idx):
        label = {k:v[idx] for k,v in self.metadata.items()}
        return self.seqlist[idx], self.hilbert_seqlist[idx], label