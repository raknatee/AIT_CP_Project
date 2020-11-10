import os
import torch
import numpy as np
import pickle

if __name__ == "__main__":

    import sys
    sys.path.append("../")
    print(sys.path)

import my_modules.util as util
from my_modules.log import log
from torch.utils.data import Dataset, DataLoader

default_y_index = [0,1]

class MyDataset(Dataset):

    def __init__(self,person_path,window_size,step_window,list_movie_index):

        super().__init__()

        with open(person_path,"rb") as f:
            self.data = pickle.load(f,encoding='bytes')
            self.data[b'labels'] = self.data[b'labels'][list_movie_index]
            self.data[b'data'] = self.data[b'data'][list_movie_index]
      
        self.window_size=window_size
        self.step_window = step_window
        self.n_per_movie = (self.data[b'data'].shape[-1]-window_size)//step_window + 1
        self.list_movie_index = list_movie_index

    def __len__(self):
        return (len(self.list_movie_index))  *(self.n_per_movie)


    def __getitem__(self,i):
        index_i = (i)//self.n_per_movie
        index_j = (i)%self.n_per_movie

        data = util.create_data_tensor_float(self.data[b'data'][index_i,:,index_j*self.step_window:index_j*self.step_window+self.window_size])
        label = util.create_data_tensor_float(self.data[b'labels'][index_i,default_y_index])
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
    

    
    