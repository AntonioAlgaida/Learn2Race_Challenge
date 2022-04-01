#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 21:53:37 2022

@author: antonio
"""



# path_train = r'/media/antonio/Crucial 1TB/L2R_challange/310c8337-c7cf-42d8-9850-4c77f151bfba_l2r_expert_demonstrations/l2r_expert_demonstrations/thruxton/train'
# ep = 0

# fig = plt.figure()
# viewer = fig.add_subplot(111)
# plt.ion() # Turns interactive mode on (probably unnecessary)
# fig.show() # Initially shows the figure

# for i in range(1000):
#     transition = np.load(f'/media/antonio/Crucial 1TB/L2R_challange/310c8337-c7cf-42d8-9850-4c77f151bfba_l2r_expert_demonstrations/l2r_expert_demonstrations/thruxton/train/episode_1/transitions_{i}.npz')
#     img = transition['img']
#     # plt.imshow(img)
#     plt.pause(.1) # Delay in seconds
#     viewer.clear() # Clears the previous image
#     viewer.imshow(img) # Loads the new image
#     # plt.pause(.1) # Delay in seconds
#     fig.canvas.draw() # Draws the image to the screen
   


#%%

# for i in range(len(transformed_dataset)):
#     sample = transformed_dataset[i]
#     plt.imshow(sample)
#     print(i, sample['image'].size(), sample['landmarks'].size())

#%%
from torch import nn
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from ILDataset import ILDataset
transformed_dataset = ILDataset()

from torch.utils.data import Dataset, DataLoader
from IL_NETs import IL_Net, VGG16, IL_Net_small
dataloader = DataLoader(transformed_dataset, batch_size=128,
                        shuffle=True, num_workers=11)

# net = VGG16().cuda()
# net = IL_Net().cuda()
net = IL_Net_small().cuda()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

actions = []

transformed_dataset = ILDataset(val=True)
val_dataloader = DataLoader(transformed_dataset, batch_size=128,
                        shuffle=True, num_workers=11)
    
for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0
    for i_batch, data in enumerate(tqdm(dataloader)):
        # get the inputs; data is a list of [inputs, labels]
        obs, act = data
        actions.extend(np.array(act.detach()))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(obs.cuda())
        loss = criterion(outputs, act.cuda())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print('[%d, %5d] loss: %.6f' % (epoch + 1, i_batch + 1, running_loss / len(dataloader)))

    # print('Finished Training')
    

    net.eval()
    with torch.no_grad():
        running_loss = 0.0
        for i_batch, data in enumerate(tqdm(val_dataloader)):
            # get the inputs; data is a list of [inputs, labels]
            obs, act = data
    
            # forward + backward + optimize
            outputs = net(obs.cuda())
            loss = criterion(outputs, act.cuda())
    
            # print statistics
            running_loss += loss.item()
        print('[%d, %5d] Validation loss: %.6f' %
              (epoch + 1, i_batch + 1, running_loss / len(val_dataloader)))

torch.save(net.state_dict(), 'best_IL.pth')


#%%
from sklearn import preprocessing

actions = np.reshape(actions, (-1, 2))

ste = actions[:, 0].reshape(-1, 1)
acc = actions[:, 1].reshape(-1, 1)

ste_q_transformer = preprocessing.QuantileTransformer()
ste_trans = ste_q_transformer.fit_transform(ste)

acc_q_transformer = preprocessing.QuantileTransformer()
acc_trans = acc_q_transformer.fit_transform(acc)

plt.plot(acc_trans, '.')
#%%
from pickle import dump
dump(ste_q_transformer, open('ste_scaler.pkl', 'wb'))
dump(acc_q_transformer, open('acc_scaler.pkl', 'wb'))

#%%
# evaluate the histogram
values, base = np.histogram(ste, bins=40)
#evaluate the cumulative
cumulative = np.cumsum(values)
# plot the cumulative function
plt.plot(base[:-1], values, c='blue')
