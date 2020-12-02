
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from my_modules.log import log
import gc 
from my_modules.model.helper import *
from my_modules.model.fft import to_fft_2
from my_modules.model.lstm2cnn_helper import transform2d

from my_modules.model.helper import move_to_gpu



class LSTM2CNN(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1,n_class=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size = 1
        self.n_class = n_class



        self.lstm_channel_size = input_size
        self.cnn1d_channel_size = input_size*2

     
        # hint : input_size =  channel

        # self.cnn_layer = [input_size,100,60,50,40,30,20,20]

        # 1450 is channel according to seq_len/window_size
        self.cnn2d_layer = [1450,500,500,300,250]
        self.cnn_layer = [self.cnn1d_channel_size,100,50,40,10,10]


        
        

        for e in range(self.n_class):
            for i in range(0,len(self.cnn_layer)-1):
                setattr(self,f'cnn{i}_c{e}',nn.Conv1d(self.cnn_layer[i],self.cnn_layer[i+1],3,stride=1))
                setattr(self,f'cnn_LR{i}_c{e}',nn.LeakyReLU())
                setattr(self,f'cnn_pooling{i}_c{e}',nn.AvgPool1d(2))
            setattr(self,f'cnn_fc_c{e}',
                nn.Sequential(
                nn.Linear(680,1000),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1000,500),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Linear(500,out_size),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                )
            )
            for i in range(0,len(self.cnn2d_layer)-1):
                setattr(self,f'cnn2d{i}_c{e}',nn.Conv2d(self.cnn2d_layer[i],self.cnn2d_layer[i+1],(3,3),stride=1))
                setattr(self,f'cnn2d_LR{i}_c{e}',nn.LeakyReLU())
                # setattr(self,f'cnn_before_lstm_pooling{i}_c{e}',nn.AvgPool2d(2))

            setattr(self,f'lstm_c{e}',
                nn.LSTM(self.lstm_channel_size,hidden_size,num_layers=number_layers,dropout=0)
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
            


        
    def forward(self,seq,norm_config=None):
        
        seq_len = seq.shape[2]
        batch = seq.shape[0]
        input_size = seq.shape[1]

        input_for_cnn_2d = transform2d(seq)
        input_for_cnn_2d = move_to_gpu(input_for_cnn_2d)
        cnn_all_output = []
        lstm_all_output = []
        for c_index in range(self.n_class):


            output_after_cnn_2d = input_for_cnn_2d
            for i in range(0,len(self.cnn2d_layer)-1):
                output_after_cnn_2d = getattr(self,f'cnn2d{i}_c{c_index}')(output_after_cnn_2d)
                output_after_cnn_2d = getattr(self,f'cnn2d_LR{i}_c{c_index}')(output_after_cnn_2d)
       
                # output_after_cnn_for_lstm = getattr(self,f'cnn_before_lstm_pooling{i}_c{c_index}')(output_after_cnn_for_lstm)


  

            lstm_input = seq.view(seq_len,batch,input_size)
           
            lstm_out,h = getattr(self,f'lstm_c{c_index}')(lstm_input,(getattr(self,f'h_c{c_index}'),getattr(self,f'c_c{c_index}')))
            setattr(self,f'h_c{c_index}',h)
            lstm_output = lstm_out[-1,:,-self.lstm_channel_size:]

            cnn_output = to_fft_2(seq)

            for i in range(0,len(self.cnn_layer)-1):
            
                cnn_output = getattr(self,f'cnn{i}_c{c_index}')(cnn_output)
                cnn_output = getattr(self,f'cnn_LR{i}_c{c_index}')(cnn_output)
                cnn_output = getattr(self,f'cnn_pooling{i}_c{c_index}')(cnn_output)

            output_after_cnn_2d = output_after_cnn_2d.flatten(start_dim =1).reshape(batch,-1)
       
            cnn_output = cnn_output.flatten(start_dim =1).reshape(batch,-1)
       
           
            
            cnn_output = torch.cat((cnn_output,output_after_cnn_2d),1)
            cnn_output=getattr(self,f'cnn_fc_c{c_index}')(cnn_output)
            lstm_output = getattr(self,f'lstm_fc_c{c_index}')(lstm_output)
            cnn_all_output.append(cnn_output)
            lstm_all_output.append(lstm_output)
  
        # v = self.final_fc_valence_no_l2(torch.cat( (cnn_output[:,0].reshape(-1,1),lstm_output[:,0].reshape(-1,1)),1  ))
        # a = self.final_fc_arousal_no_l2(torch.cat( (cnn_output[:,1].reshape(-1,1),lstm_output[:,1].reshape(-1,1)),1))
        # pred = torch.cat((v,a),1)
        last_input_feed = torch.cat((*cnn_all_output,*lstm_all_output),1)
        pred = self.final_fc_no_l2(last_input_feed)

  

        return pred


class LSTM2CNN_2(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1,n_class=2):
        super().__init__()
        self.hidden_size = 50
        self.number_layers = number_layers
        self.out_size = out_size = 1
        self.n_class = n_class



        self.lstm_channel_size = input_size*2
        self.cnn1d_channel_size = input_size*2

     
        # hint : input_size =  channel

        # self.cnn_layer = [input_size,100,60,50,40,30,20,20]

        # 1450 is channel according to seq_len/window_size
        self.cnn2d_layer = [1450,500,500,300,250]
        self.cnn_layer = [self.cnn1d_channel_size,100,50,40,10,10]


        
        

        for e in range(self.n_class):
            for i in range(0,len(self.cnn_layer)-1):
                setattr(self,f'cnn{i}_c{e}',nn.Conv1d(self.cnn_layer[i],self.cnn_layer[i+1],3,stride=1))
                setattr(self,f'cnn_LR{i}_c{e}',nn.LeakyReLU())
                setattr(self,f'cnn_pooling{i}_c{e}',nn.AvgPool1d(2))
            setattr(self,f'cnn_fc_c{e}',
                nn.Sequential(
                nn.Linear(680,1000),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1000,500),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Linear(500,out_size),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                )
            )
            for i in range(0,len(self.cnn2d_layer)-1):
                setattr(self,f'cnn2d{i}_c{e}',nn.Conv2d(self.cnn2d_layer[i],self.cnn2d_layer[i+1],(3,3),stride=1))
                setattr(self,f'cnn2d_LR{i}_c{e}',nn.LeakyReLU())
                # setattr(self,f'cnn_before_lstm_pooling{i}_c{e}',nn.AvgPool2d(2))

            setattr(self,f'lstm_c{e}',
                nn.LSTM(self.lstm_channel_size,self.hidden_size,num_layers=number_layers,dropout=0)
            )
       
            setattr(self,f'lstm_fc_c{e}',
                 nn.Sequential(
                    nn.Linear(self.hidden_size,500),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.3),
                    nn.Linear(500,200),
                    nn.Dropout(p=0.3),
                    nn.LeakyReLU(),
                    nn.Linear(200,out_size),
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
            


        
    def forward(self,seq,norm_config=None):
        
        seq_len = seq.shape[2]
        batch = seq.shape[0]
        input_size = seq.shape[1]

        input_for_cnn_2d = transform2d(seq)
        input_for_cnn_2d = move_to_gpu(input_for_cnn_2d)
        input_fft_1d = to_fft_2(seq)

        cnn_all_output = []
        lstm_all_output = []
        for c_index in range(self.n_class):


            output_after_cnn_2d = input_for_cnn_2d
            for i in range(0,len(self.cnn2d_layer)-1):
                output_after_cnn_2d = getattr(self,f'cnn2d{i}_c{c_index}')(output_after_cnn_2d)
                output_after_cnn_2d = getattr(self,f'cnn2d_LR{i}_c{c_index}')(output_after_cnn_2d)
       
                # output_after_cnn_for_lstm = getattr(self,f'cnn_before_lstm_pooling{i}_c{c_index}')(output_after_cnn_for_lstm)


  
            lstm_input = input_fft_1d
            lstm_input = lstm_input.view(seq_len,batch,-1)
           
            lstm_out,h = getattr(self,f'lstm_c{c_index}')(lstm_input,(getattr(self,f'h_c{c_index}'),getattr(self,f'c_c{c_index}')))
            setattr(self,f'h_c{c_index}',h)
            lstm_output = lstm_out[-1,:,-self.lstm_channel_size:]

            cnn_output = input_fft_1d

            for i in range(0,len(self.cnn_layer)-1):
            
                cnn_output = getattr(self,f'cnn{i}_c{c_index}')(cnn_output)
                cnn_output = getattr(self,f'cnn_LR{i}_c{c_index}')(cnn_output)
                cnn_output = getattr(self,f'cnn_pooling{i}_c{c_index}')(cnn_output)

            output_after_cnn_2d = output_after_cnn_2d.flatten(start_dim =1).reshape(batch,-1)
       
            cnn_output = cnn_output.flatten(start_dim =1).reshape(batch,-1)
       
           
            
            cnn_output = torch.cat((cnn_output,output_after_cnn_2d),1)
            cnn_output=getattr(self,f'cnn_fc_c{c_index}')(cnn_output)
            lstm_output = getattr(self,f'lstm_fc_c{c_index}')(lstm_output)
            cnn_all_output.append(cnn_output)
            lstm_all_output.append(lstm_output)
  
        # v = self.final_fc_valence_no_l2(torch.cat( (cnn_output[:,0].reshape(-1,1),lstm_output[:,0].reshape(-1,1)),1  ))
        # a = self.final_fc_arousal_no_l2(torch.cat( (cnn_output[:,1].reshape(-1,1),lstm_output[:,1].reshape(-1,1)),1))
        # pred = torch.cat((v,a),1)
        last_input_feed = torch.cat((*cnn_all_output,*lstm_all_output),1)
        pred = self.final_fc_no_l2(last_input_feed)

  

        return pred

