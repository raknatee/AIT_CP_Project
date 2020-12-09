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


class LSTMCNN_1(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size
     
        # hint : input_size =  channel

        self.cnn_layer = [input_size,40,20,20,10]
        
        for i in range(0,len(self.cnn_layer)-1):
            setattr(self,f'cnn{i}',nn.Conv1d(self.cnn_layer[i],self.cnn_layer[i+1],3,stride=2))
            setattr(self,f'cnn_LR{i}',nn.LeakyReLU())
            setattr(self,f'cnn_pooling{i}',nn.AvgPool1d(2))

                

        cnn_output_size = 20
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers=number_layers,dropout=0)
        self.fc = nn.Sequential(
          nn.Linear(hidden_size+cnn_output_size,300),
          nn.ReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(300,100),
          nn.Dropout(p=0.3),
          nn.ReLU(),
          nn.Linear(100,out_size),
          nn.Dropout(p=0.3),
          nn.ReLU(),
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
       
        lstm_output = lstm_out[-1,:,-self.hidden_size:]

        cnn_output = seq

        for i in range(0,len(self.cnn_layer)-1):
            cnn_output = getattr(self,f'cnn{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_LR{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_pooling{i}')(cnn_output)
          
        cnn_output=cnn_output.reshape((batch,-1))
        # print('cnn_out',cnn_output.shape)
        # print('lstm_output',lstm_output.shape)
        
    
        output_for_fc = torch.cat((cnn_output,lstm_output),1)
        # print('output_for_fc',output_for_fc.shape)
     
        pred = self.fc(output_for_fc)

        return pred



class LSTMCNN_2(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size
     
        # hint : input_size =  channel

        self.cnn_layer = [input_size,40,20,20,10]
        
        for i in range(0,len(self.cnn_layer)-1):
            setattr(self,f'cnn{i}',nn.Conv1d(self.cnn_layer[i],self.cnn_layer[i+1],3,stride=2))
            setattr(self,f'cnn_LR{i}',nn.LeakyReLU())
            setattr(self,f'cnn_pooling{i}',nn.AvgPool1d(2))

                

        cnn_output_size = 20
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers=number_layers,dropout=0)
        self.fc = nn.Sequential(
          nn.Linear(hidden_size+cnn_output_size,300),
          nn.LeakyReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(300,200),
          nn.Dropout(p=0.3),
          nn.LeakyReLU(),
          nn.Linear(200,100),
          nn.Dropout(p=0.3),
          nn.LeakyReLU(),
          nn.Linear(100,out_size),
          nn.Dropout(p=0.3),
          nn.LeakyReLU(),
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
       
        lstm_output = lstm_out[-1,:,-self.hidden_size:]

        cnn_output = seq

        for i in range(0,len(self.cnn_layer)-1):
            cnn_output = getattr(self,f'cnn{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_LR{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_pooling{i}')(cnn_output)
          
        cnn_output=cnn_output.reshape((batch,-1))
        # print('cnn_out',cnn_output.shape)
        # print('lstm_output',lstm_output.shape)
        
    
        output_for_fc = torch.cat((cnn_output,lstm_output),1)
        # print('output_for_fc',output_for_fc.shape)
     
        pred = self.fc(output_for_fc)

        return pred

class LSTMCNN_3(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size
     
        # hint : input_size =  channel

        self.cnn_layer = [input_size,40,20,10,4]
        
        for i in range(0,len(self.cnn_layer)-1):
            setattr(self,f'cnn{i}',nn.Conv1d(self.cnn_layer[i],self.cnn_layer[i+1],3,stride=2))
            setattr(self,f'cnn_LR{i}',nn.LeakyReLU())
            setattr(self,f'cnn_pooling{i}',nn.AvgPool1d(2))

                

        cnn_output_size = 20
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers=number_layers,dropout=0)
        self.fc = nn.Sequential(
          nn.Linear(hidden_size+cnn_output_size,300),
          nn.LeakyReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(300,200),
          nn.Dropout(p=0.3),
          nn.LeakyReLU(),
          nn.Linear(200,100),
          nn.Dropout(p=0.3),
          nn.LeakyReLU(),
          nn.Linear(100,out_size),
          nn.Dropout(p=0.3),
          nn.LeakyReLU(),
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
       
        lstm_output = lstm_out[-1,:,-self.hidden_size:]

        cnn_output = seq

        for i in range(0,len(self.cnn_layer)-1):
            cnn_output = getattr(self,f'cnn{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_LR{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_pooling{i}')(cnn_output)
          
        cnn_output=cnn_output.reshape((batch,-1))
        # print('cnn_out',cnn_output.shape)
        # print('lstm_output',lstm_output.shape)
        
    
        output_for_fc = torch.cat((cnn_output,lstm_output),1)
        # print('output_for_fc',output_for_fc.shape)
     
        pred = self.fc(output_for_fc)

        return pred



class LSTMCNN_4(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size
     
        # hint : input_size =  channel

        self.cnn_layer = [input_size,40,30,20,10]
        
        for i in range(0,len(self.cnn_layer)-1):
            setattr(self,f'cnn{i}',nn.Conv1d(self.cnn_layer[i],self.cnn_layer[i+1],3,stride=1))
            setattr(self,f'cnn_LR{i}',nn.LeakyReLU())
            setattr(self,f'cnn_pooling{i}',nn.AvgPool1d(2))

        self.cnn_fc = nn.Sequential(
          nn.Linear(880,400),
          nn.LeakyReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(400,200),
          nn.Dropout(p=0.3),
          nn.LeakyReLU(),
          nn.Linear(200,out_size),
          nn.Dropout(p=0.3),
          nn.LeakyReLU(),
        )

        self.lstm = nn.LSTM(input_size,hidden_size,num_layers=number_layers,dropout=0)
        self.lstm_fc = nn.Sequential(
          nn.Linear(hidden_size,300),
          nn.LeakyReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(300,100),
          nn.Dropout(p=0.3),
          nn.LeakyReLU(),
          nn.Linear(100,out_size),
          nn.Dropout(p=0.3),
          nn.LeakyReLU(),
        )

        self.final_fc = nn.Sequential(
          nn.Linear(4,2),
          nn.ReLU(),
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
       
        lstm_output = lstm_out[-1,:,-self.hidden_size:]

        cnn_output = seq

        for i in range(0,len(self.cnn_layer)-1):
            cnn_output = getattr(self,f'cnn{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_LR{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_pooling{i}')(cnn_output)
          
        cnn_output=self.cnn_fc(cnn_output.reshape( (batch,-1) ))
        lstm_output = self.lstm_fc(lstm_output)
        # print('output_for_fc',output_for_fc.shape)
      

        pred = self.final_fc(torch.cat((cnn_output,lstm_output),1))

        return pred


class LSTMCNN_5(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size
     
        # hint : input_size =  channel

        self.cnn_layer = [input_size,40,30,20,10]
        
        for i in range(0,len(self.cnn_layer)-1):
            setattr(self,f'cnn{i}',nn.Conv1d(self.cnn_layer[i],self.cnn_layer[i+1],3,stride=1))
            setattr(self,f'cnn_LR{i}',nn.LeakyReLU())
            setattr(self,f'cnn_pooling{i}',nn.AvgPool1d(2))

        self.cnn_fc = nn.Sequential(
          nn.Linear(880,400),
          nn.LeakyReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(400,200),
          nn.Dropout(p=0.3),
          nn.LeakyReLU(),
          nn.Linear(200,out_size),
          nn.Dropout(p=0.3),
          nn.LeakyReLU(),
        )

        self.lstm = nn.LSTM(input_size,hidden_size,num_layers=number_layers,dropout=0)
        self.lstm_fc = nn.Sequential(
          nn.Linear(hidden_size,300),
          nn.LeakyReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(300,100),
          nn.Dropout(p=0.3),
          nn.LeakyReLU(),
          nn.Linear(100,out_size),
          nn.Dropout(p=0.3),
          nn.LeakyReLU(),
        )

        self.final_fc_no_l2 = nn.Sequential(
          nn.Linear(4,2),
          nn.ReLU(),
        )

        self.reset_hidden()

    def parameters_with_l2(self):
        for name,param in self.named_parameters():
            if('no_l2' not in name):
                yield param
    def parameters_without_l2(self):
        for name,param in self.named_parameters():
            if('no_l2' in name):
                yield param
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
       
        lstm_output = lstm_out[-1,:,-self.hidden_size:]

        cnn_output = seq

        for i in range(0,len(self.cnn_layer)-1):
            cnn_output = getattr(self,f'cnn{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_LR{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_pooling{i}')(cnn_output)
          
        cnn_output=self.cnn_fc(cnn_output.reshape( (batch,-1) ))
        lstm_output = self.lstm_fc(lstm_output)
        # print('output_for_fc',output_for_fc.shape)
      

        pred = self.final_fc_no_l2(torch.cat((cnn_output,lstm_output),1))

        return pred