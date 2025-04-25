import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as functional
from utils import *
from ESMgenerator import ESMgenerator
from tqdm import tqdm
import time

class CustomSequenceDataset(Dataset):
    """
    A dataset to convert gcamp protein well fluorescence readouts into dF, rise, and decay measures.
    Meant to work with pytorch dataloaders.

    Attributes:
        metadata (dict): metadata dict of tensors, each length of dataset from the following: id, variant, set, date, plate, well, mCherry, sequence, nAP, f0, dF, rise, decay.
        rawsequences (list of string): list of strings of sequences using single letter AA abbreviations and '-' for alignment indels 
        seqdict (dict[string, tensor]): 'sequences': self.seqdict[self.rawsequences[idx]].squeeze().float(), 
        esm_seqdict (dict[string, tensor]): 'esm_sequences': self.esm_seqdict[self.rawsequences[idx]].squeeze().float(),
        hilbert_sequences (dict[string, tensor]): 'hilbert_sequences': self.hilbert_seqdict[self.rawsequences[idx]].squeeze().unsqueeze(0).float(), 
        esm_hilbert_seqdict (dict[string, tensor]): 'esm_hilbert_sequences': self.esm_hilbert_seqdict

    Methods:
        __len__(): returns number of sequences in dataset.
        __getitem__(idx): gets sequence and metadata at idx.
    """
    def __init__(self, data_file='data/train_gcamp3+6+8_90_well_mean_metrics_flattened.npz', sequence_length=450, saved_dataloader=None, dimensions=2, datarange=100, nAP_cond = None):
        
        try:
            dataloader_dict = torch.load(saved_dataloader, weights_only=True)
        except Exception as e:
            if saved_dataloader is not None:
                raise e
            print("No existing dataset passed, loading full dataset. This will take up to 5 minutes per thousand, and 1hr for the full dataset.")
            
            self.datarange = datarange
            self.sequence_length = sequence_length
            self.p = (int(np.ceil(np.sqrt(self.sequence_length)))-1).bit_length()
            self.p = self.p if self.p else 1
            self.dimensions = dimensions
            self.data = np.load(data_file)

            self.missingidx, self.datadict = self.remove_nans_and_cond(data_field='nAP' if nAP_cond else None, cond=nAP_cond) # TODO do something about this
            
            self.metadata = {}
            self.seqdict = {}
            self.rawsequences = self.gen_rawsequences(datarange)
            self.seqdict = self.gen_seqdict()
            self.metadata = self.gen_seq_and_metadata(datarange)

            self.gen = ESMgenerator()
            self.esm_seqdict = self.gen_esm_seq_list()
            
            self.hilbert_seqdict = self.gen_hilbert_seq_list()

            self.esm_hilbert_seqdict = self.gen_esm_hilbert_seq_list()

            self.datarange = len(self.metadata['id'])
            self.save_dataloader_dict(self.datarange)

        else:
            self.load_dataloader_dict(data_file, dataloader_dict)


    def remove_nans_and_cond(self, data_field='nAP', cond=None, missingidx = None): # requires self.data
        # >missing data
        print("Filtering data...")
        if missingidx is None:
            missingidx = np.isnan(self.data['dF'])
            missingidx = np.logical_or(missingidx, np.isnan(self.data['rise']))
            missingidx = np.logical_or(missingidx, np.isnan(self.data['decay']))
            if cond: missingidx = np.logical_or(missingidx, np.not_equal(self.data[data_field], cond))
            missingidx = torch.from_numpy(missingidx)
        # self.missingidx = missingidx
        datadict = {}
        for i in self.data:
            datadict[i] = self.data[i][~missingidx]
        return missingidx, datadict
    
    def gen_rawsequences(self, datarange): # requires self.datadict, self.sequence_length
        rawsequences = [str(seq) for seq in self.datadict['sequence'][0:datarange]]
        for idx, seq in enumerate(rawsequences):
            if len(seq) < self.sequence_length:
                rawsequences[idx] = seq[0:1] + "--" + seq[1:7] + "--------------------------" + seq[7:]
        print(f"Shape of rawsequences: {type(rawsequences)} of {type(rawsequences[0])} len {len(rawsequences)}")
        return rawsequences
    
    def gen_seqdict(self): # requires self.rawsequences
        seqdict = {}
        for seq in self.rawsequences:
            if seq in seqdict:
                pass
            else:
                seqdict[seq] = torch.FloatTensor([ord(x) for x in seq]).bfloat16().squeeze().unsqueeze(1)
        print(f"Shape of seqdict: {type(seqdict)} of {len(seqdict[seq])}")
        return seqdict

    def gen_seq_and_metadata(self, datarange): # requires self.datadict 
        metadata = {}
        print("Generating metadata...")
        for file in self.datadict.keys():
            if file == 'variant':
                metadata[file] = torch.FloatTensor(np.asarray(self.datadict[file], dtype=float)).bfloat16().squeeze().unsqueeze(1)[0:datarange]
            elif file == 'sequence':
                pass
            else:
                metadata[file] = torch.FloatTensor(self.datadict[file]).bfloat16().squeeze().unsqueeze(1)[0:datarange]
        print(f"Shape of metadata: {type(metadata)} len {len(metadata)}")
        return metadata

    def gen_esm_seq_list(self): # requires self.rawsequences, self.gen
        # Not actually sure why the 2-protein batch size works so well in ESM, might be related: https://arxiv.org/html/2501.07747v1
        # Generating ESM embeddings: 1 seq at a time takes 1.5s ea, 2 seq 1.5s ea, 4 seq 2.4s, 5 seq 2.8s, 10 seq 3.4s
        print("Generating ESM seq list, no curve...")
        esm_seqdict = {}
        for seq in tqdm(self.rawsequences):
            if seq in esm_seqdict:
                pass
            else:
                esm_seqdict[seq] = self.gen.residueEmbed(seq).bfloat16()
        print(f"Finished esm_seqdict")
        return esm_seqdict

    def gen_hilbert_seq_list(self): # requires self.rawsequences, and self.seqdict
        print("Generating Hilbert curved sequences...")
        hilbert_seqdict = {}
        for seq in tqdm(self.rawsequences):
            if seq in hilbert_seqdict:
                pass
            else:
                hilbert_seqdict[seq] = hilbertCurveAnyD(self.seqdict[seq].unsqueeze(0), dim=1, pad=0)
        print(f"Finished hilbert_seqdict")
        return hilbert_seqdict

    def gen_esm_hilbert_seq_list(self): # requires self.esm_seqlist and self.rawsequences
        print("Generating Hilbert curved ESM sequences...")
        esm_hilbert_seqdict = {}
        for seq in tqdm(self.rawsequences):
            if seq in esm_hilbert_seqdict:
                pass
            else:
                esm_hilbert_seqdict[seq] = hilbertCurveAnyD(self.esm_seqdict[seq], dim=1, pad=0)

        print(f"Finished esm_hilbert_seqdict")
        return esm_hilbert_seqdict
    
    def save_dataloader_dict(self, name='no_range'): # depends on everything
        print("saving dataloader dict")
        dataloader_dict = {}
        dataloader_dict['datarange'] = self.datarange
        dataloader_dict['sequence_length'] = self.sequence_length
        dataloader_dict['p'] = self.p
        dataloader_dict['dimensions'] = self.dimensions
        # dataloader_dict['data'] = self.data
        dataloader_dict['missingidx'] = self.missingidx
        # # dataloader_dict['datadict'] = self.datadict
        # # dataloader_dict['rawsequences'] = self.rawsequences
        dataloader_dict['metadata'] = self.metadata
        dataloader_dict['seqdict'] = self.seqdict
        # # dataloader_dict['gen'] = self.gen
        dataloader_dict['esm_seqdict'] = self.esm_seqdict
        dataloader_dict['hilbert_seqdict'] = self.hilbert_seqdict
        dataloader_dict['esm_hilbert_seqdict'] = self.esm_hilbert_seqdict
        torch.save(dataloader_dict, f'data/dataloader_dict{name}.pth')
        print("dataloader dict saved")

    def load_dataloader_dict(self, data_file, dataloader_dict):
        print("loading dataloader dict")
        self.data                   = np.load(data_file)
        self.datarange              = dataloader_dict['datarange']
        self.sequence_length        = dataloader_dict['sequence_length']
        self.p                      = dataloader_dict['p']
        self.dimensions             = dataloader_dict['dimensions']
        # self.data                 = dataloader_dict['data']
        self.missingidx             = dataloader_dict['missingidx']
        self.datadict               = self.remove_nans_and_cond(missingidx=self.missingidx)[1]
        self.rawsequences           = self.gen_rawsequences(self.datarange)
        self.metadata               = dataloader_dict['metadata']
        self.seqdict                = dataloader_dict['seqdict']
        self.gen                    = None if self.dimensions <3 else ESMgenerator()
        self.esm_seqdict           = dataloader_dict['esm_seqdict']
        self.hilbert_seqdict        = dataloader_dict['hilbert_seqdict']
        self.esm_hilbert_seqdict    = dataloader_dict['esm_hilbert_seqdict']
        print("dataloader dict loaded")

    def __len__(self):
        return self.datarange

    def __getitem__(self, idx):
        returndict = {
            'raw_sequences': self.rawsequences[idx], 
            'sequences': self.seqdict[self.rawsequences[idx]].squeeze().float(), 
            'esm_sequences': self.esm_seqdict[self.rawsequences[idx]].squeeze().float(),
            'hilbert_sequences': self.hilbert_seqdict[self.rawsequences[idx]].squeeze().unsqueeze(0).float(), 
            'esm_hilbert_sequences': self.esm_hilbert_seqdict[self.rawsequences[idx]].permute(0, 3, 1, 2).squeeze().float()
        }
        for k,v in self.metadata.items():
            returndict[k] = v[idx].float()
        return returndict