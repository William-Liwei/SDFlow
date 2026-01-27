import torch
from torch.utils import data

import pandas as pd
from scipy import io
import numpy as np


class TSDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 192, unit_length = 4, dataset_type='train'):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 'stock':
            csv_path = './dataset/stock_data.csv'
        elif dataset_name == 'energy':
            csv_path = './dataset/energy_data.csv'
        elif dataset_name == 'etth':
            csv_path = './dataset/ETTh.csv'
        elif dataset_name == 'fmri':
            csv_path = './dataset/sim4.mat'

        if dataset_name in ['stock','energy']:
            data = pd.read_csv(csv_path).values.astype(float)
        elif dataset_name == 'fmri':
            data = io.loadmat(csv_path)['ts']
        else:
            data = pd.read_csv(csv_path).values[:,1:].astype(float)

        num_train = int(len(data) * 1.0)# less than 1 for prediction tasks
        num_test = int(len(data) * 0.0)
        num_vali = len(data) - num_train - num_test

        border1s = [0, num_train, len(data) - num_test]
        border2s = [num_train, num_train + num_vali, len(data)]
        train_data = data[border1s[0]:border2s[0]]

        self.mean = train_data.mean(0)
        self.std = train_data.std(0)
        self.min = train_data.min(0)
        self.max = train_data.max(0)
        data = (data - self.min) / (self.max - self.min)


        if dataset_type == 'train':
            self.data = data[border1s[0]:border2s[0]]
        elif dataset_type == 'val':
            self.data = data[border1s[1]:border2s[1]]
        elif dataset_type == 'test':
            self.data = data[border1s[2]:border2s[2]]
        print("{} data: Length is {}, Number of nodes is {}".format(dataset_type, self.data.shape[0], self.data.shape[1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def norm_transform(self, data):
        return (data - self.mean) / self.std
    def inv_minmax_transform(self, data):
        return data * (self.max - self.min) + self.min



    def __len__(self):
        return (len(self.data) - self.window_size) + 1

    def __getitem__(self, item):
        data = self.data[item:item+self.window_size]

        return data

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 192,
               unit_length = 4,
               dataset_type='train'):

    if dataset_name == 'sine':
        data_dir = "./dataset/sine_ground_truth_24_train.npy"
        trainSet = np.load(data_dir)
    elif dataset_name == 'mujoco':
        data_dir = "./dataset/mujoco_norm_truth_24_train.npy"
        trainSet = np.load(data_dir)
    else:
        trainSet = TSDataset(dataset_name, window_size=window_size, unit_length=unit_length, dataset_type=dataset_type)


    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last = False)
    
    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
