import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from my_modules.log import log
import gc 
from my_modules.model.helper import *


do_gpu = True
def move_to_gpu(item):
    
    if(torch.cuda.is_available() and do_gpu ):
#         print('moving to GPU',torch.cuda.get_device_name(0))
        return item.cuda()
    else:
        return item


class LSTMCNN_6(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size
     
        # hint : input_size =  channel

        # self.cnn_layer = [input_size,100,60,50,40,30,20,20]
        self.cnn_layer = [input_size,40,30,20,10]

        self.jump_point= [0,3]
        self.des_point= [2,len(self.cnn_layer)-1]
        self.save_point = []
        
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

        # self.final_fc_valence_no_l2 = nn.Sequential(
        #   nn.Linear(2,1),
        #   nn.ReLU(),
        # )
        # self.final_fc_arousal_no_l2 = nn.Sequential(
        #   nn.Linear(2,1),
        #   nn.ReLU(),
        # )
        self.final_fc_no_l2 = nn.Sequential(
          nn.Linear(4,2),
        #   nn.LeakyReLU(),
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
            # if(i in self.des_point):
            #     a= cnn_output
            #     b= self.save_point[self.des_point.index(i)]
            #     cnn_output = torch.cat((a,b),1)
            cnn_output = getattr(self,f'cnn{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_LR{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_pooling{i}')(cnn_output)
            # if(i in self.jump_point):
            #     self.save_point.append(cnn_output)
          
      
        cnn_output=self.cnn_fc(cnn_output.reshape( (batch,-1) ))
        lstm_output = self.lstm_fc(lstm_output)
  
        # v = self.final_fc_valence_no_l2(torch.cat( (cnn_output[:,0].reshape(-1,1),lstm_output[:,0].reshape(-1,1)),1  ))
        # a = self.final_fc_arousal_no_l2(torch.cat( (cnn_output[:,1].reshape(-1,1),lstm_output[:,1].reshape(-1,1)),1))
        # pred = torch.cat((v,a),1)
      
        pred = self.final_fc_no_l2(torch.cat((cnn_output,lstm_output),1))

  

        return pred




class LSTMCNN_7(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size
     
        # hint : input_size =  channel

        # self.cnn_layer = [input_size,100,60,50,40,30,20,20]
        self.cnn_layer = [input_size,40,30,20,10]

        self.jump_point= [0,3]
        self.des_point= [2,len(self.cnn_layer)-1]
        self.save_point = []
        
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
          nn.Linear(4,4),
          nn.ReLU(),
          nn.Linear(4,4),
          nn.ReLU(),
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
            # if(i in self.des_point):
            #     a= cnn_output
            #     b= self.save_point[self.des_point.index(i)]
            #     cnn_output = torch.cat((a,b),1)
            cnn_output = getattr(self,f'cnn{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_LR{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_pooling{i}')(cnn_output)
            # if(i in self.jump_point):
            #     self.save_point.append(cnn_output)
          
      
        cnn_output=self.cnn_fc(cnn_output.reshape( (batch,-1) ))
        lstm_output = self.lstm_fc(lstm_output)
  

        pred = self.final_fc_no_l2(torch.cat((cnn_output,lstm_output),1))

  

        return pred


class LSTMCNN_8(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size
     
        # hint : input_size =  channel

        # self.cnn_layer = [input_size,100,60,50,40,30,20,20]
        self.cnn_layer = [input_size,40,30,20,10]

        self.jump_point= [0,3]
        self.des_point= [2,len(self.cnn_layer)-1]
        self.save_point = []
        
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

        # self.final_fc_valence_no_l2 = nn.Sequential(
        #   nn.Linear(2,1),
        #   nn.ReLU(),
        # )
        # self.final_fc_arousal_no_l2 = nn.Sequential(
        #   nn.Linear(2,1),
        #   nn.ReLU(),
        # )
        self.final_fc_no_l2 = nn.Sequential(
          nn.Linear(4,2),
          nn.LeakyReLU(),
        #   nn.ReLU(),

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
            # if(i in self.des_point):
            #     a= cnn_output
            #     b= self.save_point[self.des_point.index(i)]
            #     cnn_output = torch.cat((a,b),1)
            cnn_output = getattr(self,f'cnn{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_LR{i}')(cnn_output)
            cnn_output = getattr(self,f'cnn_pooling{i}')(cnn_output)
            # if(i in self.jump_point):
            #     self.save_point.append(cnn_output)
          
      
        cnn_output=self.cnn_fc(cnn_output.reshape( (batch,-1) ))
        lstm_output = self.lstm_fc(lstm_output)
  
        # v = self.final_fc_valence_no_l2(torch.cat( (cnn_output[:,0].reshape(-1,1),lstm_output[:,0].reshape(-1,1)),1  ))
        # a = self.final_fc_arousal_no_l2(torch.cat( (cnn_output[:,1].reshape(-1,1),lstm_output[:,1].reshape(-1,1)),1))
        # pred = torch.cat((v,a),1)
      
        pred = self.final_fc_no_l2(torch.cat((cnn_output,lstm_output),1))

  

        return pred


class LSTMCNN_9(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1,n_class=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size = 1
        self.n_class = n_class
     
        # hint : input_size =  channel

        # self.cnn_layer = [input_size,100,60,50,40,30,20,20]
        self.cnn_layer = [input_size,40,30,20,10]


        
        

        for e in range(self.n_class):
            for i in range(0,len(self.cnn_layer)-1):
                setattr(self,f'cnn{i}_c{e}',nn.Conv1d(self.cnn_layer[i],self.cnn_layer[i+1],3,stride=1))
                setattr(self,f'cnn_LR{i}_c{e}',nn.LeakyReLU())
                setattr(self,f'cnn_pooling{i}_c{e}',nn.AvgPool1d(2))
            setattr(self,f'cnn_fc_c{e}',
                nn.Sequential(
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
            )
            setattr(self,f'lstm_c{e}',
                nn.LSTM(input_size,hidden_size,num_layers=number_layers,dropout=0)
            )
       
            setattr(self,f'lstm_fc_c{e}',
                 nn.Sequential(
                    nn.Linear(hidden_size,300),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.3),
                    nn.Linear(300,100),
                    nn.Dropout(p=0.3),
                    nn.LeakyReLU(),
                    nn.Linear(100,out_size),
                    nn.Dropout(p=0.3),
                    nn.LeakyReLU(),))
       


        self.final_fc_no_l2 = nn.Sequential(
          nn.Linear(4,2),
          nn.LeakyReLU(),
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

        for i in range(self.n_class):
            setattr(self,f'h_c{i}',torch.zeros(arg_0,bz,self.hidden_size))
            setattr(self,f'c_c{i}',torch.zeros(arg_0,bz,self.hidden_size))
         
            setattr(self,f'h_c{i}',move_to_gpu(getattr(self,f'h_c{i}') ))
            setattr(self,f'c_c{i}',move_to_gpu(getattr(self,f'c_c{i}') ))
            


        
    def forward(self,seq):

        seq_len = seq.shape[2]
        batch = seq.shape[0]
        input_size = seq.shape[1]

        cnn_all_output = []
        lstm_all_output = []
        for c_index in range(self.n_class):
            lstm_out,h = getattr(self,f'lstm_c{c_index}')(seq.view(seq_len,batch,input_size),(getattr(self,f'h_c{c_index}'),getattr(self,f'c_c{c_index}')))
            setattr(self,f'h_c{c_index}',h)
            lstm_output = lstm_out[-1,:,-self.hidden_size:]

            cnn_output = seq

            for i in range(0,len(self.cnn_layer)-1):
            
                cnn_output = getattr(self,f'cnn{i}_c{c_index}')(cnn_output)
                cnn_output = getattr(self,f'cnn_LR{i}_c{c_index}')(cnn_output)
                cnn_output = getattr(self,f'cnn_pooling{i}_c{c_index}')(cnn_output)
        
            
        
            cnn_output=getattr(self,f'cnn_fc_c{c_index}')(cnn_output.reshape( (batch,-1) ))
            lstm_output = getattr(self,f'lstm_fc_c{c_index}')(lstm_output)
            cnn_all_output.append(cnn_output)
            lstm_all_output.append(lstm_output)
  
        # v = self.final_fc_valence_no_l2(torch.cat( (cnn_output[:,0].reshape(-1,1),lstm_output[:,0].reshape(-1,1)),1  ))
        # a = self.final_fc_arousal_no_l2(torch.cat( (cnn_output[:,1].reshape(-1,1),lstm_output[:,1].reshape(-1,1)),1))
        # pred = torch.cat((v,a),1)
        last_input_feed = torch.cat((*cnn_all_output,*lstm_all_output),1)
        pred = self.final_fc_no_l2(last_input_feed)

  

        return pred

