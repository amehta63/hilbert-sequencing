import torch
import torch.nn.functional as functional
import torch.optim as optim

import time

from models import *
from utils import *


def train_model(model = HilbertClassifier(),
    epochs=1,
    train_data=None,
    log_int=100,
    save_int=None,
    save_pth=f"models/default_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.pth",
    test_data=None,
    print_logs = True,
    device = 'cpu'
):
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn1 = functional.mse_loss
    
    model.to(device=device)
    model.train() #good habit, unnecessary
        
    logs = []
    start = time.time()
    base = start
    
    for epoch in range(epochs):
        model.train()
        for index, (sequence, metadata) in enumerate(train_data):
            with torch.set_grad_enabled(True):

                p = (int(np.ceil(np.sqrt(len(sequence[0]))))-1).bit_length()

                sequence = functional.pad(sequence, (0, 2**(2*p)-len(sequence[0]))).to(device=device)
                twoDseq = sequence.clone()
                twoDseq = twoDseq.reshape(-1, 2**p, 2**p)
                for i in range(twoDseq.shape[0]):
                    twoDseq[i] = hilbertCurve(sequence[i]).squeeze()
                twoDseq = twoDseq.unsqueeze(1)

                optimizer.zero_grad()
                outputlabel = model(twoDseq)
                loss1 = loss_fn1(outputlabel, metadata['dF'])
                loss = loss1
                loss.backward()
                optimizer.step()
                

            if save_int is not None:
                if not (index+1) % save_int:
                    save_model(model, save_pth, epoch, index, loss)

            logs.append(loss.detach())

            if log_int is not None:
                if not index % log_int:
                    if print_logs: print(f"Epoch: {epoch}, Batch: {index}/{len(train_data)}, Loss: {loss}, time: {time.time() - start}")
                    start = time.time()
                
    print(f"Total time: {time.time() - base}")
    save_model(model, save_pth, epoch, index, loss)
    
    return model, logs
