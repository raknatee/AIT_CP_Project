import torch

def create_data_tensor_float(x):
    return torch.tensor(x,dtype=torch.float32)

def create_data_tensor_long(x):
    return torch.tensor(x,dtype=torch.long)
