import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from my_modules.log import log
import gc 


do_gpu = True
def move_to_gpu(item):
    
    if(torch.cuda.is_available() and do_gpu ):
#         print('moving to GPU',torch.cuda.get_device_name(0))
        return item.cuda()
    else:
        return item
    
def del_gpu(*args):
    # print("before del gpu",torch.cuda.memory_allocated())
    for arg in args:
        arg.cpu()
        del arg
    gc.collect()
    torch.cuda.empty_cache()
    # print("after del gpu",torch.cuda.memory_allocated())
    
def get_used_memory():
    return torch.cuda.memory_allocated()



class LSTM_0(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers=number_layers)

        self.reset_hidden()
    
    def reset_hidden(self):
        arg_0 = self.number_layers
        
        self.h = torch.zeros(arg_0,1,self.hidden_size)
        self.c = torch.zeros(arg_0,1,self.hidden_size)
        
        self.h = move_to_gpu(self.h)
        self.c = move_to_gpu(self.c)
        
    def forward(self,seq):

        seq_len = seq.shape[2]
        batch = seq.shape[0]
        input_size = seq.shape[1]


        
        lstm_out,self.hidden = self.lstm(seq.view(seq_len,batch,input_size),(self.h,self.c))
        # log(lstm_out.shape)
        pred = lstm_out[-1,-1,-self.out_size:].reshape(self.out_size)

        return pred


class LSTM_1(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers=number_layers)
        self.linear = nn.Sequential(
          nn.Linear(2,10),
          nn.ReLU(),
          nn.Linear(10,2),
          nn.ReLU()
        )

        self.reset_hidden()
    
    def reset_hidden(self):
        arg_0 = self.number_layers
        
        self.h = torch.zeros(arg_0,1,self.hidden_size)
        self.c = torch.zeros(arg_0,1,self.hidden_size)
        
        self.h = move_to_gpu(self.h)
        self.c = move_to_gpu(self.c)
        
    def forward(self,seq):

        seq_len = seq.shape[2]
        batch = seq.shape[0]
        input_size = seq.shape[1]


        
        lstm_out,self.hidden = self.lstm(seq.view(seq_len,batch,input_size),(self.h,self.c))
        # log(lstm_out.shape)
        pred = lstm_out[-1,-1,-self.out_size:].reshape(self.out_size)
        pred = self.linear(pred)

        return pred

class LSTM_2(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers=number_layers)
        self.linear = nn.Sequential(
          nn.Linear(hidden_size,hidden_size//2),
          nn.ReLU(),
          nn.Linear(hidden_size//2,out_size),
          nn.ReLU()
        )

        self.reset_hidden()
    
    def reset_hidden(self):
        arg_0 = self.number_layers
        
        self.h = torch.zeros(arg_0,1,self.hidden_size)
        self.c = torch.zeros(arg_0,1,self.hidden_size)
        
        self.h = move_to_gpu(self.h)
        self.c = move_to_gpu(self.c)
        
    def forward(self,seq):

        seq_len = seq.shape[2]
        batch = seq.shape[0]
        input_size = seq.shape[1]


        
        lstm_out,self.hidden = self.lstm(seq.view(seq_len,batch,input_size),(self.h,self.c))
        # log(lstm_out.shape)
        pred = lstm_out[-1,-1,-self.hidden_size:].reshape(self.hidden_size)
        pred = self.linear(pred)

        return pred


class LSTM_3(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers=number_layers,dropout=0)
        self.linear = nn.Sequential(
          nn.Linear(hidden_size,hidden_size//2),
          nn.ReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(hidden_size//2,out_size),
          nn.Dropout(p=0.3),
          nn.ReLU()
        )

        self.reset_hidden()
    
    def reset_hidden(self):
        arg_0 = self.number_layers
        
        self.h = torch.zeros(arg_0,1,self.hidden_size)
        self.c = torch.zeros(arg_0,1,self.hidden_size)
        
        self.h = move_to_gpu(self.h)
        self.c = move_to_gpu(self.c)
        
    def forward(self,seq):

        seq_len = seq.shape[2]
        batch = seq.shape[0]
        input_size = seq.shape[1]


        
        lstm_out,self.hidden = self.lstm(seq.view(seq_len,batch,input_size),(self.h,self.c))
        # log(lstm_out.shape)
        pred = lstm_out[-1,-1,-self.hidden_size:].reshape(self.hidden_size)
        pred = self.linear(pred)

        return pred


class LSTM_4(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size
     
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers=number_layers,dropout=0)
        self.fc = nn.Sequential(
          nn.Linear(hidden_size,hidden_size*3),
          nn.ReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(hidden_size*3,out_size),
          nn.Dropout(p=0.3),
          nn.ReLU()
        )

        self.reset_hidden()
    
    def reset_hidden(self,set_batch_size=None):
        arg_0 = self.number_layers
        bz = None
        if(set_batch_size is not None):
            bz = set_batch_size
        else:
            bz = 1
        self.h = torch.zeros(arg_0,bz,self.hidden_size)
        self.c = torch.zeros(arg_0,bz,self.hidden_size)
        
        self.h = move_to_gpu(self.h)
        self.c = move_to_gpu(self.c)
        
    def forward(self,seq):

        seq_len = seq.shape[2]
        batch = seq.shape[0]
        input_size = seq.shape[1]


        
        lstm_out,self.hidden = self.lstm(seq.view(seq_len,batch,input_size),(self.h,self.c))
       
        next_feed = lstm_out[-1,:,-self.hidden_size:]
    
   
        pred = self.fc(next_feed)

        return pred

