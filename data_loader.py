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
        sequence (tensor): list of sequences, all zero_padded to 450 amino acids long.

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
            print("No existing dataset passed, loading full dataset. This will take up to 20 minutes for 2D and 5 hours for 3D.")
            
            self.datarange = datarange
            self.sequence_length = sequence_length
            self.p = (int(np.ceil(np.sqrt(self.sequence_length)))-1).bit_length()
            self.p = self.p if self.p else 1
            self.dimensions = dimensions
            self.data = np.load(data_file)

            self.missingidx, self.datadict = self.remove_nans_and_cond(data_field='nAP' if nAP_cond else None, cond=nAP_cond) # TODO do something about this
            
            self.metadata = {}
            self.seqdict = {}
            self.seqlist = [] 
            self.rawsequences, self.seqlist, self.seqdict = self.gen_raw_and_seqlist(datarange)
            # self.rawsequences, self.metadata, self.seqlist = self.gen_seq_and_metadata(datarange)
            self.metadata = self.gen_seq_and_metadata(datarange)

            if self.dimensions > 2:
                self.gen = ESMgenerator()
                self.esm_seqlist, self.esm_seq_dict = self.gen_esm_seq_list(datarange)
            
            self.hilbert_seqlist, self.hilbert_seqdict = self.gen_hilbert_seq_list(datarange)

            if self.dimensions > 2: 
                self.esm_hilbert_seqlist, self.esm_hilbert_seqdict = self.gen_esm_hilbert_seq_list(datarange)
            else: self.esm_hilbert_seqlist = self.hilbert_seqlist

            datarange = len(self.metadata['id'])
            self.save_dataloader_dict(datarange)

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
    
    def gen_raw_and_seqlist(self, datarange):
        seqdict = {}
        seqlist = []
        rawsequences = self.datadict['sequence'][0:datarange]
        for seq in self.datadict["sequence"][0:datarange]:
            if seq in seqdict:
                seqlist.append(seqdict[seq])
            else:
                seqdict[seq] = functional.pad(torch.FloatTensor([ord(x) for x in seq]).bfloat16(), pad=(0, self.sequence_length-len(seq))).squeeze().unsqueeze(1)
                seqlist.append(seqdict[seq]) 
        seqlist = torch.hstack(seqlist).T
        print(f"Shape of sequence data and metadata: rawsequences: {type(rawsequences)} of {type(rawsequences[0])} len {len(rawsequences)}, seqlist: {seqlist.shape}")
        return rawsequences, seqlist, seqdict

    def gen_seq_and_metadata(self, datarange): # requires both self.datadict and self.sequence_length
        # rawsequences = []
        metadata = {}
        # seqlist = []
        print("Generating metadata...")
        # rawsequences = self.datadict['sequence'][0:datarange]
        for file in self.datadict.keys():
            if file == 'variant':
                metadata[file] = torch.FloatTensor(np.asarray(self.datadict[file], dtype=float)).bfloat16().squeeze().unsqueeze(1)[0:datarange]
            elif file == 'sequence':
                pass
                # for seq in self.datadict[file][0:datarange]:
                #     seqlist.append(functional.pad(torch.FloatTensor([ord(x) for x in seq]).bfloat16(), pad=(0, self.sequence_length-len(seq))).squeeze().unsqueeze(1))
            else:
                metadata[file] = torch.FloatTensor(self.datadict[file]).bfloat16().squeeze().unsqueeze(1)[0:datarange]
        # seqlist = torch.hstack(seqlist).T
        # print(f"Shape of sequence data and metadata: rawsequences: {type(rawsequences)} of {type(rawsequences[0])} len {len(rawsequences)}, metadata: {type(metadata)} len {len(metadata)}, seqlist: {seqlist.shape}")
        print(f"Shape of metadata: {type(metadata)} len {len(metadata)}")
        # return rawsequences, metadata, seqlist
        return metadata

    def gen_esm_seq_list(self, datarange): # requires self.rawsequences, self.gen, self.datarange
        # Not actually sure why the 2-protein batch size works so well in ESM, might be related: https://arxiv.org/html/2501.07747v1
        # Generating ESM embeddings: 1 seq at a time takes 1.5s ea, 2 seq 1.5s ea, 4 seq 2.4s, 5 seq 2.8s, 10 seq 3.4s
        # esm_seqlist = self.gen.listOfResidueEmbed(self.rawsequences, layer=33) # takes about as long as batch-2
        # esm_seqlist = self.gen.rawListOfResidueEmbed(self.rawsequences) # the slowest option, takes more RAM than I have
        print("Generating ESM seq list, no curve...")
        esm_seq_dict = {}
        esm_seqlist = []
        # for i in tqdm(range(int(self.datarange/2))): # decided to keep this even though its ugly code, the tqdm progress bar is nice
        #     esm_seqlist.append(self.gen.rawListOfResidueEmbed(self.rawsequences[2*i:2*i+2]).bfloat16())
        # if self.datarange % 2:
        #     esm_seqlist.append(self.gen.rawListOfResidueEmbed(self.rawsequences[-1]).bfloat16())
        for seq in tqdm(self.rawsequences):
            if seq in esm_seq_dict:
                esm_seqlist.append(esm_seq_dict[seq])
            else:
                esm_seq_dict[seq] = self.gen.residueEmbed(seq).bfloat16()
                esm_seqlist.append(esm_seq_dict[seq])
        esm_seqlist =  torch.cat(esm_seqlist, dim=0).bfloat16()
        print(f"Shape of ESM seq list: {esm_seqlist.shape}")
        return esm_seqlist, esm_seq_dict

    def gen_hilbert_seq_list(self, datarange): # requires self.p, self.sequence_length, and self.seqlist
        print("Generating Hilbert curved sequences...")
        hilbert_seqlist2 = [] # DELETE ME
        hilbert_seqdict = {}
        for seq in tqdm(self.rawsequences):
            if seq in hilbert_seqdict:
                hilbert_seqlist2.append(hilbert_seqdict[seq])
            else:
                hilbert_seqdict[seq] = hilbertCurveAnyD(self.seqdict[seq].unsqueeze(0), dim=1, pad=0)
                hilbert_seqlist2.append(hilbert_seqdict[seq])
        hilbert_seqlist2 =  torch.cat(hilbert_seqlist2, dim=0).bfloat16().squeeze().unsqueeze(1)

        hilbert_seqlist = hilbertCurveAnyD(self.seqlist[0:datarange].unsqueeze(2), dim=1, pad=0)
        hilbert_seqlist = hilbert_seqlist.squeeze().unsqueeze(1)
        print(f"Shape of Hilbert curved sequences: {hilbert_seqlist.shape}")
        return hilbert_seqlist, hilbert_seqdict

    def gen_esm_hilbert_seq_list(self, datarange): # requires self.esm_seqlist
        esm_hilbert_seqlist = []
        print("Generating Hilbert curved ESM sequences...")
        esm_hilbert_seqlist2 = [] # DELETE ME
        esm_hilbert_seqdict = {}
        for seq in tqdm(self.rawsequences):
            if seq in esm_hilbert_seqdict:
                esm_hilbert_seqlist2.append(esm_hilbert_seqdict[seq])
            else:
                esm_hilbert_seqdict[seq] = hilbertCurveAnyD(self.esm_seq_dict[seq], dim=1, pad=0)
                esm_hilbert_seqlist2.append(esm_hilbert_seqdict[seq])
        esm_hilbert_seqlist2 =  torch.cat(esm_hilbert_seqlist2, dim=0).bfloat16().permute(0, 3, 1, 2)

        esm_hilbert_seqlist = hilbertCurveAnyD(self.esm_seqlist, dim=1, pad=0).permute(0, 3, 1, 2)
        print(torch.equal(esm_hilbert_seqlist, esm_hilbert_seqlist2)) # DELETE ME
        print(f"Shape of Hilbert curved ESM sequences: {esm_hilbert_seqlist.shape}")
        return esm_hilbert_seqlist, esm_hilbert_seqdict
    
    def save_dataloader_dict(self, name='no_range'): # depends on everything
        print("saving dataloader dict")
        dataloader_dict = {}
        dataloader_dict['datarange'] = self.datarange
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
        print("loading dataloader dict")
        self.data                   = np.load(data_file)
        self.datarange              = dataloader_dict['datarange']
        self.sequence_length        = dataloader_dict['sequence_length']
        self.p                      = dataloader_dict['p']
        self.dimensions             = dataloader_dict['dimensions']
        # self.data                 = dataloader_dict['data']
        self.missingidx             = dataloader_dict['missingidx']
        self.datadict               = self.remove_nans_and_cond(missingidx=self.missingidx)[1] # TODO do something about this ## dataloader_dict['datadict']
        self.rawsequences           = self.datadict['sequence'][0:self.datarange]
        self.metadata               = dataloader_dict['metadata']
        self.seqlist                = dataloader_dict['seqlist']
        self.gen                    = None if self.dimensions <3 else ESMgenerator()
        self.esm_seqlist            = dataloader_dict['esm_seqlist']
        self.hilbert_seqlist        = dataloader_dict['hilbert_seqlist']
        self.esm_hilbert_seqlist    = dataloader_dict['esm_hilbert_seqlist']
        print("dataloader dict loaded")

    def __len__(self):
        return self.datarange #len(self.hilbert_seqlist)

    def __getitem__(self, idx):
        # returndict = { # TODO fix these names
        #     'raw_seqlist': self.rawsequences[idx], 
        #     'seqlist': self.seqlist[idx].float(), 
        #     'esm_seqlist': self.esm_seqlist[idx].float(),
        #     'hilbert_seqlist': self.hilbert_seqlist[idx].float(), 
        #     'esm_hilbert_seqlist': self.esm_hilbert_seqlist[idx].float()
        # }
        returndict = { # TODO fix these names
            'raw_seqlist': self.rawsequences[idx], 
            'seqlist': self.seqdict[self.rawsequences[idx]].squeeze().float(), 
            'esm_seqlist': self.esm_seq_dict[self.rawsequences[idx]].squeeze().float(),
            'hilbert_seqlist': self.hilbert_seqdict[self.rawsequences[idx]].squeeze().unsqueeze(0).float(), 
            'esm_hilbert_seqlist': self.esm_hilbert_seqdict[self.rawsequences[idx]].permute(0, 3, 1, 2).squeeze().float()
        }
        for k,v in self.metadata.items():
            returndict[k] = v[idx].float()
        return returndict