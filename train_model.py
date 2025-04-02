import torch
import torch.nn.functional as functional
import torch.optim as optim

import time

from models import *
from utils import *


def train_model(model = HilbertClassifier(),
    epochs=1,
    train_data=None,
    dimensions='2D',
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
    substart = start
    
    for epoch in range(epochs):
        model.train()
        for index, (sequence, twoDseq, threeDseq, metadata) in enumerate(train_data):
            # print(f"for loop time: {time.time() - substart}"); substart = time.time()
            sequence = sequence.to(device)
            if dimensions == '2D': input = twoDseq.to(device)
            if dimensions == '3D': input = threeDseq.to(device)
            for md in metadata.keys():
                metadata[md] = metadata[md].to(device)

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
                    if print_logs: print(f"Epoch: {epoch}, Batch: {index}/{len(train_data)}, Loss: {loss1:.2f} dF + {loss2:.2f} rise + {loss3:.2f} decay = {loss:.2f}, time: {time.time() - start:.2f}s")
                    start = time.time()
                
    print(f"Total time: {time.time() - base:.2f}s")
    save_model(model, save_pth, epoch, index, loss)
    
    return model, logs
