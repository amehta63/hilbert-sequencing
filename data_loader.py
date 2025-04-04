import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as functional
from utils import *
from ESMgenerator import ESMgenerator
from tqdm import tqdm


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
    def __init__(self, data_file='data/train_gcamp3+6+8_90_well_mean_metrics_flattened.npz', sequence_length=450, saved_dataloader=None, dimensions=2, range=100, nAP_cond = None):
        
        try:
            dataloader_dict = torch.load(saved_dataloader, weights_only=True)
        except Exception as e:
            if saved_dataloader is not None:
                raise e
            print("No existing dataset passed, loading full dataset. This will take up to 20 minutes for 2D and 5 hours for 3D.")
            self.sequence_length = sequence_length
            self.dimensions = dimensions
            datadict = {}
            self.data = np.load(data_file)
            missingidx = np.isnan(self.data['dF'])
            missingidx = np.logical_or(missingidx, np.isnan(self.data['rise']))
            missingidx = np.logical_or(missingidx, np.isnan(self.data['decay']))
            if nAP_cond: missingidx = np.logical_or(missingidx, np.not_equal(self.data['nAP'], nAP_cond))
            self.missingidx = missingidx

            for i in self.data:
                datadict[i] = self.data[i][~self.missingidx]

            self.data = datadict

            self.metadata = {}
            self.seqlist = []
            self.asciilist = []
            print("Generating sequence data and metadata...")
            self.rawsequences = self.data['sequence'][0:range]
            for file in self.data.keys():
                if file == 'variant':
                    self.metadata[file] = torch.FloatTensor(np.asarray(self.data[file], dtype=float)).squeeze().unsqueeze(1)
                elif file == 'sequence':
                    for seq in self.data[file]:
                        self.seqlist.append(functional.pad(torch.FloatTensor([ord(x) for x in seq]), pad=(0, self.sequence_length-len(seq))).squeeze().unsqueeze(1))
                else:
                    self.metadata[file] = torch.FloatTensor(self.data[file]).squeeze().unsqueeze(1)
            self.seqlist = torch.hstack(self.seqlist).T

            if self.dimensions > 2: 
                print("Generating ESM seq list, no curve...")
                self.gen = ESMgenerator()
                # self.ESMlist = self.gen.listOfResidueEmbed(self.rawsequences, layer=33)
                self.ESMlist = []
                for seq in tqdm(self.rawsequences):
                    residue_embedding = self.gen.residueEmbed(seq, layer=33).unsqueeze(0)
                    self.ESMlist.append(residue_embedding)
                self.ESMlist =  torch.cat(self.ESMlist, dim=0)

            self.p = (int(np.ceil(np.sqrt(self.sequence_length)))-1).bit_length()

            print("Generating Hilbert curved sequences...")
            self.hilbert_seqlist = []
            for seq in tqdm(self.seqlist[0:range]):
                sequence = functional.pad(seq, (0, 2**(2*self.p)-self.sequence_length)) # make them all the same length
                twoDseq = hilbertCurve1Dto2D(sequence).squeeze() # make them into pxp matricies
                # twoDseq = twoDseq.reshape(-1, 2**self.p, 2**self.p) # make them 1xpxp tensors TODO: this should be unsqueeze
                twoDseq = twoDseq.unsqueeze(0)
                self.hilbert_seqlist.append(twoDseq)
            self.hilbert_seqlist = torch.cat(self.hilbert_seqlist, dim=0).unsqueeze(1)

            self.esm_hilbert_seqlist = []
            if self.dimensions > 2: 
                print("Generating Hilbert curved ESM sequences...")
                for seq in tqdm(self.data['sequence'][0:range]):
                    self.esm_hilbert_seqlist.append(hilbertCurve2Dto3D(self.gen.residueEmbed(seq)).unsqueeze(0))
                self.esm_hilbert_seqlist = torch.cat(self.esm_hilbert_seqlist, dim=0).permute(0, 3, 1, 2)
            else: self.esm_hilbert_seqlist = self.hilbert_seqlist

            dataloader_dict = {}
            dataloader_dict['sequence_length'] = self.sequence_length
            dataloader_dict['dimensions'] = self.dimensions
            # dataloader_dict['data'] = self.data
            dataloader_dict['missingidx'] = torch.from_numpy(self.missingidx)
            dataloader_dict['metadata'] = self.metadata
            dataloader_dict['seqlist'] = self.seqlist
            dataloader_dict['p'] = self.p
            dataloader_dict['ESMlist'] = self.ESMlist
            dataloader_dict['hilbert_seqlist'] = self.hilbert_seqlist
            dataloader_dict['esm_hilbert_seqlist'] = self.esm_hilbert_seqlist
            torch.save(dataloader_dict, f'data/dataloader_dict{range}.pth')
        else:
            self.data = np.load(data_file)
            self.rawsequences = self.data['sequence']
            self.sequence_length = dataloader_dict['sequence_length']
            self.dimensions = dataloader_dict['dimensions']
            #self.data = dataloader_dict['data']
            self.missingidx = dataloader_dict['missingidx']
            self.metadata = dataloader_dict['metadata']
            self.seqlist = dataloader_dict['seqlist']
            self.p = dataloader_dict['p']
            self.ESMlist = dataloader_dict['ESMlist']
            self.hilbert_seqlist = dataloader_dict['hilbert_seqlist']
            self.esm_hilbert_seqlist = dataloader_dict['esm_hilbert_seqlist']

    def __len__(self):
        return len(self.hilbert_seqlist)

    def __getitem__(self, idx):
        returndict = { # TODO fix these names
            'raw_seqlist': self.rawsequences[idx], 
            'seqlist': self.seqlist[idx], 
            'esm_seqlist': self.ESMlist[idx],
            'hilbert_seqlist': self.hilbert_seqlist[idx], 
            'esm_hilbert_seqlist': self.esm_hilbert_seqlist[idx]
        }
        for k,v in self.metadata.items():
            returndict[k] = v[idx] 
        return returndict