import numpy as np
import torch

from my_modules.model.helper import move_to_gpu

def fft(signal,delta_time):
    fourier_transform = np.fft.fft(signal)/len(signal)
    fourier_transform = np.abs(fourier_transform[:fourier_transform.size])
#     fourier_transform = np.abs(fourier_transform[:fourier_transform.size//2])
    frequency_series = np.arange(fourier_transform.shape[0])/delta_time
    return frequency_series,fourier_transform
def to_fft(data):

    fs = 8064//55
    ds = 1/fs
    returned = torch.zeros_like(data)
    batch_size = returned.shape[0]
    channel_size = returned.shape[1]
    for b in range(batch_size):
        for c in range(channel_size):
            returned[b,c] = move_to_gpu(torch.tensor(fft(data[b,c].cpu(),ds)[1]) )
   
    return returned

def to_fft_2(data_set):
    output = torch.rfft(data_set,1,onesided =False)
    output = (output[:,:,:,0]**2+output[:,:,:,1]**2)**(0.5)
    return torch.cat((output,data_set),1 )


