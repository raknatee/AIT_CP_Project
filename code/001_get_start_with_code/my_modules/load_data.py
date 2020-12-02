import os
import torch
import numpy as np
import pickle
import random

if __name__ == "__main__":

    import sys
    sys.path.append("../")
    print(sys.path)

import my_modules.util as util
from my_modules.log import log
from torch.utils.data import Dataset, DataLoader

default_y_index = [0,1]


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.len_of_dataset = [len(d) for d in self.datasets]

    def __getitem__(self, index):

        which_dataset = 0
        index_in_dataset = index
        for index_of_len,len_ in enumerate(self.len_of_dataset):
            if(index_in_dataset<len_):
                which_dataset = index_of_len
                break
            else:
                index_in_dataset -= len_
        return self.datasets[which_dataset][index_in_dataset]

    def __len__(self):
        return sum(len(d) for d in self.datasets)

class MyDataset(Dataset):

    def __init__(self,person_path,window_size,step_window,list_movie_index,do_noise=False):

        super().__init__()

        with open(person_path,"rb") as f:
            self.data = pickle.load(f,encoding='bytes')
            self.data[b'labels'] = self.data[b'labels'][list_movie_index]
            self.data[b'data'] = self.data[b'data'][list_movie_index]
            self.mean = (self.data[b'data'].mean(axis=(0,2))).reshape(1,-1,1)
            self.std = (self.data[b'data'].std(axis=(0,2))).reshape(1,-1,1)
        self.do_noise = do_noise
        self.window_size=window_size
        self.step_window = step_window
        self.n_per_movie = (self.data[b'data'].shape[-1]-window_size)//step_window + 1
        self.list_movie_index = list_movie_index
    def norm_param(self):
        return {'mean':self.mean,'std':self.std}

    def __len__(self):
        return (len(self.list_movie_index))  *(self.n_per_movie)


    def __getitem__(self,i):

            
        index_i = (i)//self.n_per_movie
        index_j = (i)%self.n_per_movie

        data = util.create_data_tensor_float(self.data[b'data'][index_i,:,index_j*self.step_window:index_j*self.step_window+self.window_size])
        label = util.create_data_tensor_float(self.data[b'labels'][index_i,default_y_index])

        if(self.do_noise and (random.random()<0.7)  ):
            noise_param = self.norm_param()
            noise_param['mean'] = torch.squeeze(torch.FloatTensor(noise_param['mean']))
            noise_param['std'] = torch.squeeze(torch.FloatTensor(noise_param['std']))
         
            noise_mean = (noise_param['mean'])/15
            noise_std = (noise_param['std'])/15
    
            for i in range(noise_mean.shape[0]):

                noise = torch.normal(mean=noise_mean[i],
                        std=noise_std[i],size=(1,data.shape[-1]) )
                data[i] += torch.squeeze(noise)
            
        


        return (data,label)

# test_case
if __name__ == "__main__":

    d = MyDataset(r'C:\Users\rakna\Desktop\AIT_working\AIT_CP_Project\this_folder_git_ignore\ori_datasets\s01.dat',
    145*10,20,[i for i in range(38)]
    )
    print(len([*d]))
    print(len(d))
    # print(d[0])
    # print(d[3024])
    # print(d[3025])
    print('d[0] data',d[798][0][0].shape )
    # print('d[0] label',d[0].shape )
    

    
    