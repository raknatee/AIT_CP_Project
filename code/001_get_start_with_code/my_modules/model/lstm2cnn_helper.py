from my_modules.model.mapping_matrix import eeg_channels as eeg_channel_map
import torch
def transform2d(seq):
    seq_len = seq.shape[2]
    batch_size = seq.shape[0]
    h_and_w = len(eeg_channel_map)
    returned_tensor = torch.zeros((batch_size,h_and_w,h_and_w,seq_len))
    
    for batch_index in range(batch_size):
        for i in range(h_and_w):
            for j in range(h_and_w):
                if(eeg_channel_map[i][j] != -1):
                    index_channel = eeg_channel_map[i][j] 
                    returned_tensor[batch_index][i][j] = seq[batch_index,index_channel,:] 
    
    returned_tensor = returned_tensor.reshape((batch_size,seq_len,h_and_w,h_and_w))
    return returned_tensor




