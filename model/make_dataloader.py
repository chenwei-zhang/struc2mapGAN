import sys
sys.path.append('/home/cwzhang/project/struc2mapGAN')

import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from utils.map2cube import create_cube



class FakeDataset(Dataset):
    def __init__(self, num_samples=100, img_size=(32, 32, 32), batch_size=32, shuffle=True, num_workers=8):
        self.num_samples = num_samples
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # Generate fake data
        np.random.seed(0)
        self.imgs = np.random.rand(self.num_samples, *self.img_size)
        np.random.seed(42)
        self.labels = np.random.rand(self.num_samples, *self.img_size)

        # Convert to PyTorch tensors
        self.imgs = torch.from_numpy(self.imgs).float().unsqueeze(dim=1)
        self.labels = torch.from_numpy(self.labels).float().unsqueeze(dim=1)
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError('Index out of range')
        return self.imgs[idx], self.labels[idx]

    def create_dataloader(self):
        dataset = self
        train_dataset, val_dataset = random_split(dataset, [80, 20], generator=torch.Generator().manual_seed(42))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_dataloader, val_dataloader
    
    

class GAN_Train_Dataset(Dataset):
    """
    Input: simulated high-resolution maps: '{emd_id}_sim_norm.mrc'
    Target: experimental high-resolution maps: '{emd_id}_norm.mrc'
    """
    def __init__(self, train_dir, cube_text):
        self.train_dir = train_dir
        self.cube_text = cube_text
        
        with open(f'{self.cube_text}/train_gan_cubes.txt') as f:
            self.lines = f.readlines()
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        cb_name = self.lines[idx].strip('\n')
        load_data = np.load(f'{self.train_dir}/{cb_name}', 'r')
        input = load_data['input_cube']
        target = load_data['target_cube']
        # convert to tensor and add channel dimension
        input = torch.from_numpy(input).float().unsqueeze(dim=0)
        target = torch.from_numpy(target).float().unsqueeze(dim=0)
        
        return input, target
        
        

class GAN_Val_Dataset(Dataset):
    """
    Input: simulated high-resolution maps: '{emd_id}_sim_norm.mrc'
    Target: experimental high-resolution maps: '{emd_id}_norm.mrc'
    """
    def __init__(self, val_dir, cube_text):
        self.val_dir = val_dir
        self.cube_text = cube_text
        
        with open(f'{self.cube_text}/val_gan_cubes.txt') as f:
            self.lines = f.readlines()
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        cb_name = self.lines[idx].strip('\n')
        load_data = np.load(f'{self.val_dir}/{cb_name}', 'r')
        input = load_data['input_cube']
        target = load_data['target_cube']
        # convert to tensor and add channel dimension
        input = torch.from_numpy(input).float().unsqueeze(dim=0)
        target = torch.from_numpy(target).float().unsqueeze(dim=0)
        
        return input, target      
    
    
    
class GAN_Pred_Dataset(Dataset):
    def __init__(self, map):
        self.map = map
        self.cube = np.array(create_cube(self.map, box_size=32, core_size=20))
        # convert to tensor and add channel
        self.cube = torch.from_numpy(self.cube).float().unsqueeze(dim=1)
        
    def __len__(self):
        return len(self.cube)
    
    def __getitem__(self, idx):
        return self.cube[idx]
