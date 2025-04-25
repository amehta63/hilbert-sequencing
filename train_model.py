import torch
import torch.nn.functional as functional
import torch.optim as optim

import time
from models import *
from utils import *


def train_model(model = HilbertClassifier2D.HilbertClassifier2D(),
    epochs=1,
    train_data=None,
    dimensions='2D',
    log_int=100,
    save_int=None,
    save_pth=f"checkpoints/default_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.pth",
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
    substart = start
    
    for epoch in range(epochs):
        model.train()
        for index, seqdict in enumerate(train_data):
            # print(f"for loop time: {time.time() - substart}"); substart = time.time()
            if dimensions == '1D': input = seqdict['sequences'].to(device)
            if dimensions == '1DESM': input = seqdict['esm_sequences'].to(device)
            if dimensions == '2D': input = seqdict['hilbert_sequences'].to(device)
            if dimensions == '3D': input = seqdict['esm_hilbert_sequences'].to(device)
            metadata = {}
            for md in ['id', 'variant', 'set', 'date', 'plate', 'well', 'mCherry', 'nAP', 'f0', 'dF', 'rise', 'decay']:
                metadata[md] = seqdict[md].to(device)

            with torch.set_grad_enabled(True):

                optimizer.zero_grad()
                dF, rise, decay = model(input)
                # print(f"model time: {time.time() - substart}"); substart = time.time()

                loss1 = loss_fn1(dF, metadata['dF'])
                loss2 = loss_fn1(rise, metadata['rise'])
                loss3 = loss_fn1(decay, metadata['decay'])
                loss = loss1 + loss2 + loss3
                loss.backward()
                optimizer.step()

                # print(f"loss time: {time.time() - substart}"); substart = time.time()
                

            if save_int is not None:
                if not (index+1) % save_int:
                    save_model(model, save_pth, epoch, index, loss)

            logs.append(loss.detach())

            if log_int is not None:
                if not index % log_int:
                    if print_logs: print(f"Epoch: {epoch}, Batch: {index}/{len(train_data)}, Loss: {loss1:.3f} dF + {loss2:.3f} rise + {loss3:.3f} decay = {loss:.3f}, time: {time.time() - start:.3f}s")
                    start = time.time()
                
    print(f"Total time: {time.time() - base:.3f}s")
    save_model(model, save_pth, epoch, index, loss)
    
    return model, logs
