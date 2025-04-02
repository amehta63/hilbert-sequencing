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
    def __init__(self, model_path='esm2_t33_650M_UR50D'):
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
        token_representations = self.tokenEmbed(sequence, layer)["representations"][layer]
        return token_representations[0, 1:len(sequence)+1]
    
    def residueTaggedEmbed(self, sequence, layer=33):
        return torch.cat((sequence, self.residueEmbed(sequence, layer)), 1) #TODO fix this