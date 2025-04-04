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
            
            self.range = range
            self.sequence_length = sequence_length
            self.p = (int(np.ceil(np.sqrt(self.sequence_length)))-1).bit_length()
            self.dimensions = dimensions
            self.data = np.load(data_file)

            self.missingidx, self.datadict = self.remove_nans_and_cond(data_field='nAP' if nAP_cond else None, cond=nAP_cond) # TODO do something about this
            
            self.metadata = {}
            self.seqlist = []
            self.rawsequences, self.metadata, self.seqlist = self.gen_seq_and_metadata(range)

            if self.dimensions > 2:
                self.gen = ESMgenerator()
                self.esm_seqlist = self.gen_esm_seq_list()
            
            self.hilbert_seqlist = self.gen_hilbert_seq_list(range)

            if self.dimensions > 2: self.esm_hilbert_seqlist = self.gen_esm_hilbert_seq_list(range)
            else: self.esm_hilbert_seqlist = self.hilbert_seqlist

            if range is None: range = "no_range"
            self.save_dataloader_dict(range)

        else:
            self.load_dataloader_dict(data_file, dataloader_dict)


    def remove_nans_and_cond(self, data_field='nAP', cond=None, missingidx = None): #requires self.data
        # >missing data
        print("Filtering data...")
        if missingidx is None:
            missingidx = np.isnan(self.data['dF'])
            missingidx = np.logical_or(missingidx, np.isnan(self.data['rise']))
            missingidx = np.logical_or(missingidx, np.isnan(self.data['decay']))
            if cond: missingidx = np.logical_or(missingidx, np.not_equal(self.data[data_field], cond))
        # self.missingidx = missingidx
        datadict = {}
        for i in self.data:
            datadict[i] = self.data[i][~missingidx]
        return missingidx, datadict
    
    def gen_seq_and_metadata(self, range): # requires both self.datadict and self.sequence_length
        rawsequences = []
        metadata = {}
        seqlist = []
        print("Generating sequence data and metadata...")
        rawsequences = self.datadict['sequence'][0:range]
        for file in self.datadict.keys():
            if file == 'variant':
                metadata[file] = torch.FloatTensor(np.asarray(self.datadict[file], dtype=float)).squeeze().unsqueeze(1)
            elif file == 'sequence':
                for seq in self.datadict[file][0:range]:
                    seqlist.append(functional.pad(torch.FloatTensor([ord(x) for x in seq]), pad=(0, self.sequence_length-len(seq))).squeeze().unsqueeze(1))
            else:
                metadata[file] = torch.FloatTensor(self.datadict[file]).squeeze().unsqueeze(1)
        seqlist = torch.hstack(seqlist).T
        print(f"Shape of sequence data and metadata: rawsequences: {type(rawsequences)} of {type(rawsequences[0])} len {len(rawsequences)}, metadata: {type(metadata)} len {len(metadata)}, seqlist: {seqlist.shape}")
        return rawsequences, metadata, seqlist

    def gen_esm_seq_list(self): # requires self.rawsequences and self.gen
        print("Generating ESM seq list, no curve...")
        # self.esm_seqlist = self.gen.listOfResidueEmbed(self.rawsequences, layer=33)
        esm_seqlist = []
        for seq in tqdm(self.rawsequences):
            residue_embedding = self.gen.residueEmbed(seq, layer=33).unsqueeze(0)
            esm_seqlist.append(residue_embedding)
        esm_seqlist =  torch.cat(esm_seqlist, dim=0)
        print(f"Shape of ESM seq list: {esm_seqlist.shape}")
        return esm_seqlist

    def gen_hilbert_seq_list(self, range): # requires self.p, self.sequence_length, and self.seqlist
        print("Generating Hilbert curved sequences...")
        hilbert_seqlist = []
        for seq in tqdm(self.seqlist[0:range]):
            sequence = functional.pad(seq, (0, 2**(2*self.p)-self.sequence_length)) # make them all the same length
            twoDseq = hilbertCurve1Dto2D(sequence).squeeze() # make them into pxp matricies
            # twoDseq = twoDseq.reshape(-1, 2**self.p, 2**self.p) # make them 1xpxp tensors TODO: this should be unsqueeze
            twoDseq = twoDseq.unsqueeze(0)
            hilbert_seqlist.append(twoDseq)
        hilbert_seqlist = torch.cat(hilbert_seqlist, dim=0).unsqueeze(1)
        print(f"Shape of Hilbert curved sequences: {hilbert_seqlist.shape}")
        return hilbert_seqlist

    def gen_esm_hilbert_seq_list(self, range): # requires self.gen, self.datadict
        esm_hilbert_seqlist = []
        print("Generating Hilbert curved ESM sequences...")
        for seq in tqdm(self.datadict['sequence'][0:range]):
            esm_hilbert_seqlist.append(hilbertCurve2Dto3D(self.gen.residueEmbed(seq)).unsqueeze(0))
        esm_hilbert_seqlist = torch.cat(esm_hilbert_seqlist, dim=0).permute(0, 3, 1, 2)
        print(f"Shape of Hilbert curved ESM sequences: {esm_hilbert_seqlist.shape}")
        return esm_hilbert_seqlist
    
    def save_dataloader_dict(self, name=range): # depends on everything
        print("saving dataloader dict")
        dataloader_dict = {}
        dataloader_dict['range'] = self.range
        dataloader_dict['sequence_length'] = self.sequence_length
        dataloader_dict['p'] = self.p
        dataloader_dict['dimensions'] = self.dimensions
        # dataloader_dict['data'] = self.data
        dataloader_dict['missingidx'] = torch.from_numpy(self.missingidx)
        # dataloader_dict['datadict'] = self.datadict
        # dataloader_dict['rawsequences'] = self.rawsequences
        dataloader_dict['metadata'] = self.metadata
        dataloader_dict['seqlist'] = self.seqlist
        # dataloader_dict['gen'] = self.gen
        dataloader_dict['esm_seqlist'] = self.esm_seqlist
        dataloader_dict['hilbert_seqlist'] = self.hilbert_seqlist
        dataloader_dict['esm_hilbert_seqlist'] = self.esm_hilbert_seqlist
        torch.save(dataloader_dict, f'data/dataloader_dict{name}.pth')
        print("dataloader dict saved")

    def load_dataloader_dict(self, data_file, dataloader_dict):
        print("loader dataloader dict")
        self.data                   = np.load(data_file)
        self.range                  = dataloader_dict['range']
        self.sequence_length        = dataloader_dict['sequence_length']
        self.p                      = dataloader_dict['p']
        self.dimensions             = dataloader_dict['dimensions']
        # self.data                 = dataloader_dict['data']
        self.missingidx             = dataloader_dict['missingidx']
        self.datadict               = self.remove_nans_and_cond(missingidx=self.missingidx)[1] # TODO do something about this ## dataloader_dict['datadict']
        self.rawsequences           = self.datadict['sequence'][0:self.range]
        self.metadata               = dataloader_dict['metadata']
        self.seqlist                = dataloader_dict['seqlist']
        self.gen                    = None if self.dimensions <3 else ESMgenerator()
        self.esm_seqlist            = dataloader_dict['esm_seqlist']
        self.hilbert_seqlist        = dataloader_dict['hilbert_seqlist']
        self.esm_hilbert_seqlist    = dataloader_dict['esm_hilbert_seqlist']
        print("dataloader dict loaded")

    def __len__(self):
        return self.range #len(self.hilbert_seqlist)

    def __getitem__(self, idx):
        returndict = { # TODO fix these names
            'raw_seqlist': self.rawsequences[idx], 
            'seqlist': self.seqlist[idx], 
            'esm_seqlist': self.esm_seqlist[idx],
            'hilbert_seqlist': self.hilbert_seqlist[idx], 
            'esm_hilbert_seqlist': self.esm_hilbert_seqlist[idx]
        }
        for k,v in self.metadata.items():
            returndict[k] = v[idx] 
        return returndict