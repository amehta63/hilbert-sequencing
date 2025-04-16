import torch
import esm

class ESMgenerator():
    """
    A dataset to convert gcamp protein well fluorescence readouts into dF, rise, and decay measures.
    Meant to work with pytorch dataloaders.

    Attributes:
        model_path (string): model name, see: https://github.com/facebookresearch/esm?tab=readme-ov-file#pre-trained-models-.

    Methods:
        tokenEmbed(sequence, layer=33): returns dict with keys: 'logits', 'representations', 'attentions', 'contacts'.
        residueEmbed(sequence, layer=33): returns tensor of "representations" aka an embedding of size sequencelength x embedding size.
        residueTaggedEmbed(sequence, layer=33): returns the residueEmbed but with the original sequence concatenated on top - THIS DOES NOT WORK. #TODO fix this
    """
    def __init__(self, model_path='esm2_t33_650M_UR50D', device='cpu'):
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

    def tokenEmbed(self, sequence, layer=33):
        """
        Returns dict with keys: 'logits', 'representations', 'attentions', 'contacts'
        
        """
        data = [("proteinx", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            return self.model(batch_tokens, repr_layers=[layer], return_contacts=True) 
    
    def residueEmbed(self, sequence, layer=33):
        if isinstance(sequence, list):
            return self.listOfResidueEmbed(sequence, layer)
        token_representations = self.tokenEmbed(sequence, layer)["representations"][layer]
        return token_representations[0, 1:len(sequence)+1].unsqueeze(0)
    
    def listOfResidueEmbed(self, sequenceList, layer=33):
        if isinstance(sequenceList, str):
            return self.residueEmbed(sequenceList, layer)
        seqlist = []
        for seq in sequenceList:
            residue_embedding = self.residueEmbed(seq, layer)
            seqlist.append(residue_embedding)
        return torch.cat(seqlist, dim=0)
    
    def rawListOfResidueEmbed(self, rawSequenceList, layer=33):
        data = []
        for i,v in enumerate(rawSequenceList):
            data.append((f"protein{i}", v))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            full_output = self.model(batch_tokens, repr_layers=[layer], return_contacts=True) 
        token_representations = full_output["representations"][layer]
        return token_representations[:,1:-1,:]
    
    def residueEmbedfromOrd(self, sequenceList, layer=33): # this is insanely slow, please do not use
        seqlist = []
        for seq in sequenceList:
            sequence = ''.join(chr(i) for i in seq.int())
            token_representations = self.tokenEmbed(sequence, layer)["representations"][layer]
            seqlist.append(token_representations[0, 1:len(sequence)+1])
        return torch.cat(seqlist, dim=0)
    
    def residueTaggedEmbed(self, sequence, layer=33):
        return torch.cat((sequence, self.residueEmbed(sequence, layer)), 1) #TODO fix this