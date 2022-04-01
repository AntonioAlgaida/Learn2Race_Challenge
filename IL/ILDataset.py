#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:04:49 2022

@author: antonio
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import crop
from torchvision import transforms
from PIL import Image
from pickle import load

ste_q_transformer = load(open('scalers/ste_scaler.pkl', 'rb'))
acc_q_transformer = load(open('scalers/acc_scaler.pkl', 'rb'))

# def crop_center(image):
#     return crop(image, 0, 0, 800, 800)

def normalize_actions(act):
    # max_acc = 4. 
    # min_acc = -6.
    
    # max_ste = 1.0
    # min_ste = -1.0
    
    ste = act[0].reshape(-1, 1)
    ste_norm = ste_q_transformer.transform(ste)

    # ste_norm = (ste - min_ste)/(max_ste - min_ste) * 2 - 1
    
    acc = act[1].reshape(-1, 1)
    acc_norm = acc_q_transformer.transform(acc)

    # acc_norm = (acc - min_acc)/(max_acc - min_acc) * 2 - 1
    
    return np.reshape((ste_norm, acc_norm), (-1))
    # return np.reshape(acc_norm, (-1))


class ILDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, val=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        root_dir = '/media/antonio/Crucial 1TB/L2R_challange/310c8337-c7cf-42d8-9850-4c77f151bfba_l2r_expert_demonstrations/l2r_expert_demonstrations/thruxton/train/episode_0/'
        # root_dir = 'I:/L2R_challange/310c8337-c7cf-42d8-9850-4c77f151bfba_l2r_expert_demonstrations/l2r_expert_demonstrations/thruxton/train/episode_0/'
        val_dir = '/media/antonio/Crucial 1TB/L2R_challange/310c8337-c7cf-42d8-9850-4c77f151bfba_l2r_expert_demonstrations/l2r_expert_demonstrations/thruxton/val/episode_2/'
        if val:
            root_dir = val_dir
        self.val = val
        self.root_dir = root_dir
        self.list_IDs = os.listdir(root_dir)
        
        self.transformations = transforms.Compose([transforms.ToTensor(),
                                                   # transforms.ColorJitter(.5, .5, .5, .25),
                                                    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                                   # transforms.RandomAdjustSharpness(sharpness_factor=2),
                                                   # transforms.RandomAutocontrast(),
                                                   transforms.RandomInvert(),
                                                   transforms.Grayscale(),
                                                   transforms.Resize((96, 96*2)),
                                                   transforms.Normalize((0.5), (0.5))])
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        if self.val:
            idx += 6949
        try:
            transition = np.load(f'{self.root_dir}transitions_{idx}.npz')
            img = transition['img']
            img = Image.fromarray(img)
            img = crop(img, 150, 0, 125, 500)
    
            # img_tensor = torch.from_numpy(img)
            # img_tensor = img_tensor.permute(2, 0, 1)/255
            img = self.transformations(img)
    
            # img[img <= 0] = -1
            # img[img > 0] = 1
            
            action = transition['action']
            action = normalize_actions(action)
            action_tensor = torch.from_numpy(action)
            # if self.transform:
            #     img = self.transform(img)
        except:
            # print(f'Error capting {idx}')
            idx = 6950
            transition = np.load(f'{self.root_dir}transitions_{idx}.npz')
            img = transition['img']
            img = Image.fromarray(img)
            img = crop(img, 150, 0, 125, 500)

            img = self.transformations(img)
            
            # img[img <= 0] = -1
            # img[img > 0] = 1
            
            action = transition['action']
            action = normalize_actions(action)

            action_tensor = torch.from_numpy(action)
            
        return img, action_tensor.float()
