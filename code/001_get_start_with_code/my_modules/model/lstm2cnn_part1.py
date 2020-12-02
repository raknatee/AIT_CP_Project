import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from my_modules.log import log
import gc 
from my_modules.model.helper import *
from my_modules.model.lstm2cnn_helper import transform2d

do_gpu = True
def move_to_gpu(item):
    
    if(torch.cuda.is_available() and do_gpu ):
#         print('moving to GPU',torch.cuda.get_device_name(0))
        return item.cuda()
    else:
        return item

class LSTM2CNN_1(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1,n_class=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size = 1
        self.n_class = n_class
     
        # hint : input_size =  channel

        # self.cnn_layer = [input_size,100,60,50,40,30,20,20]

        # 1450 is channel according to seq_len/window_size
        self.cnn_before_lstm_layer = [1450,500,200,100]
        self.cnn_layer = [input_size,40,30,20,10]


        
        

        for e in range(self.n_class):
            for i in range(0,len(self.cnn_layer)-1):
                setattr(self,f'cnn{i}_c{e}',nn.Conv1d(self.cnn_layer[i],self.cnn_layer[i+1],3,stride=1))
                setattr(self,f'cnn_LR{i}_c{e}',nn.LeakyReLU())
                setattr(self,f'cnn_pooling{i}_c{e}',nn.AvgPool1d(2))
            setattr(self,f'cnn_fc_c{e}',
                nn.Sequential(
                nn.Linear(880,300),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(300,100),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Linear(100,out_size),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                )
            )
            for i in range(0,len(self.cnn_before_lstm_layer)-1):
                setattr(self,f'cnn_before_lstm{i}_c{e}',nn.Conv2d(self.cnn_before_lstm_layer[i],self.cnn_before_lstm_layer[i+1],(3,3),stride=1))
                setattr(self,f'cnn_before_lstm_LR{i}_c{e}',nn.LeakyReLU())
                # setattr(self,f'cnn_before_lstm_pooling{i}_c{e}',nn.AvgPool2d(2))

            
            setattr(self,f'lstm_c{e}',
                # 9 -> 3*3 last image from cnn_before_lstm
                nn.LSTM(9,hidden_size,num_layers=number_layers,dropout=0)
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

        input_for_cnn_before_lstm = transform2d(seq)
        input_for_cnn_before_lstm = move_to_gpu(input_for_cnn_before_lstm)
        if(norm_config is not None):
            mean = move_to_gpu(torch.Tensor(norm_config['mean']))
            std = move_to_gpu(torch.Tensor(norm_config['std']) )
            seq = (seq-mean)/std
        cnn_all_output = []
        lstm_all_output = []
        for c_index in range(self.n_class):


            output_after_cnn_for_lstm = input_for_cnn_before_lstm
            for i in range(0,len(self.cnn_before_lstm_layer)-1):
                output_after_cnn_for_lstm = getattr(self,f'cnn_before_lstm{i}_c{c_index}')(output_after_cnn_for_lstm)
                output_after_cnn_for_lstm = getattr(self,f'cnn_before_lstm_LR{i}_c{c_index}')(output_after_cnn_for_lstm)
       
                # output_after_cnn_for_lstm = getattr(self,f'cnn_before_lstm_pooling{i}_c{c_index}')(output_after_cnn_for_lstm)

            output_after_cnn_for_lstm = output_after_cnn_for_lstm.flatten(start_dim =2)

            ooo_batch_size = output_after_cnn_for_lstm.shape[0]
            ooo_seq_len = output_after_cnn_for_lstm.shape[1]
            output_after_cnn_for_lstm = output_after_cnn_for_lstm.reshape((ooo_seq_len,ooo_batch_size,-1))
        

            lstm_out,h = getattr(self,f'lstm_c{c_index}')(output_after_cnn_for_lstm,(getattr(self,f'h_c{c_index}'),getattr(self,f'c_c{c_index}')))
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


class LSTM2CNN_2(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1,n_class=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size = 1
        self.n_class = n_class
     
        # hint : input_size =  channel

        # self.cnn_layer = [input_size,100,60,50,40,30,20,20]
        self.cnn_layer = [input_size,40,30,20,10]
        self.cnn_2d_layer = [1450,500,200,100]



        
        

        for e in range(self.n_class):
            for i in range(0,len(self.cnn_layer)-1):
                setattr(self,f'cnn{i}_c{e}',nn.Conv1d(self.cnn_layer[i],self.cnn_layer[i+1],3,stride=1))
                setattr(self,f'cnn_LR{i}_c{e}',nn.LeakyReLU())
                setattr(self,f'cnn_pooling{i}_c{e}',nn.AvgPool1d(2))
            setattr(self,f'cnn_fc_c{e}',
                nn.Sequential(
                nn.Linear(880,300),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(300,100),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Linear(100,out_size),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                )
            )


            for i in range(0,len(self.cnn_2d_layer)-1):
                setattr(self,f'cnn_2d{i}_c{e}',nn.Conv2d(self.cnn_2d_layer[i],self.cnn_2d_layer[i+1],(3,3),stride=1))
                setattr(self,f'cnn_2d_LR{i}_c{e}',nn.LeakyReLU())
            setattr(self,f'cnn_2d_fc_c{e}',
                nn.Sequential(
                nn.Linear(900,300),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(300,100),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Linear(100,out_size),
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
          nn.Linear(6,2),
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

       
        if(norm_config is not None):
            mean = move_to_gpu(torch.Tensor(norm_config['mean']))
            std = move_to_gpu(torch.Tensor(norm_config['std']) )
            seq = (seq-mean)/std
        all_output = []
        for c_index in range(self.n_class):
            lstm_out,h = getattr(self,f'lstm_c{c_index}')(seq.view(seq_len,batch,input_size),(getattr(self,f'h_c{c_index}'),getattr(self,f'c_c{c_index}')))
            setattr(self,f'h_c{c_index}',h)
            lstm_output = lstm_out[-1,:,-self.hidden_size:]

            cnn_output = seq

            for i in range(0,len(self.cnn_layer)-1):
            
                cnn_output = getattr(self,f'cnn{i}_c{c_index}')(cnn_output)
                cnn_output = getattr(self,f'cnn_LR{i}_c{c_index}')(cnn_output)
                cnn_output = getattr(self,f'cnn_pooling{i}_c{c_index}')(cnn_output)
    
            cnn_2d_output = transform2d(seq)
            cnn_2d_output = move_to_gpu(cnn_2d_output)
            for i in range(0,len(self.cnn_2d_layer)-1):
                cnn_2d_output = getattr(self,f'cnn_2d{i}_c{c_index}')(cnn_2d_output)
                cnn_2d_output = getattr(self,f'cnn_2d_LR{i}_c{c_index}')(cnn_2d_output)
       

        
            cnn_output=getattr(self,f'cnn_fc_c{c_index}')(cnn_output.reshape( (batch,-1) ))
            lstm_output = getattr(self,f'lstm_fc_c{c_index}')(lstm_output)
            # print(cnn_2d_output.reshape( (batch,-1) ).shape)
            cnn_2d_output = getattr(self,f'cnn_2d_fc_c{c_index}')(cnn_2d_output.reshape( (batch,-1) ))
            # print(cnn_2d_output.shape)
            all_output.append(cnn_output)
            all_output.append(lstm_output)
            all_output.append(cnn_2d_output)
  
        
        last_input_feed = torch.cat(all_output,1)
        pred = self.final_fc_no_l2(last_input_feed)

  

        return pred


class LSTM2CNN_3(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1,n_class=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size = 1
        self.n_class = n_class
     
        # hint : input_size =  channel

        # self.cnn_layer = [input_size,100,60,50,40,30,20,20]

        # 1450 is channel according to seq_len/window_size
        self.cnn_before_lstm_layer = [1450,500,200,100]
        self.cnn_layer = [input_size,40,30,20,10]
        self.cnn_2d_layer = [1450,500,200,100]

        
        

        for e in range(self.n_class):
            for i in range(0,len(self.cnn_layer)-1):
                setattr(self,f'cnn{i}_c{e}',nn.Conv1d(self.cnn_layer[i],self.cnn_layer[i+1],3,stride=1))
                setattr(self,f'cnn_LR{i}_c{e}',nn.LeakyReLU())
                setattr(self,f'cnn_pooling{i}_c{e}',nn.AvgPool1d(2))
            setattr(self,f'cnn_fc_c{e}',
                nn.Sequential(
                nn.Linear(880,300),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(300,100),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Linear(100,out_size),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                )
            )
            for i in range(0,len(self.cnn_before_lstm_layer)-1):
                setattr(self,f'cnn_before_lstm{i}_c{e}',nn.Conv2d(self.cnn_before_lstm_layer[i],self.cnn_before_lstm_layer[i+1],(3,3),stride=1))
                setattr(self,f'cnn_before_lstm_LR{i}_c{e}',nn.LeakyReLU())
                # setattr(self,f'cnn_before_lstm_pooling{i}_c{e}',nn.AvgPool2d(2))

            
            setattr(self,f'lstm_c{e}',
                # 9 -> 3*3 last image from cnn_before_lstm
                nn.LSTM(9,hidden_size,num_layers=number_layers,dropout=0)
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
       
            for i in range(0,len(self.cnn_2d_layer)-1):
                setattr(self,f'cnn_2d{i}_c{e}',nn.Conv2d(self.cnn_2d_layer[i],self.cnn_2d_layer[i+1],(3,3),stride=1))
                setattr(self,f'cnn_2d_LR{i}_c{e}',nn.LeakyReLU())
            setattr(self,f'cnn_2d_fc_c{e}',
                nn.Sequential(
                nn.Linear(900,300),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(300,100),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Linear(100,out_size),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                )
            )


        self.final_fc_no_l2 = nn.Sequential(
          nn.Linear(6,4),
          nn.LeakyReLU(),
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

        input_for_cnn_before_lstm = transform2d(seq)
        input_for_cnn_before_lstm = move_to_gpu(input_for_cnn_before_lstm)
        if(norm_config is not None):
            mean = move_to_gpu(torch.Tensor(norm_config['mean']))
            std = move_to_gpu(torch.Tensor(norm_config['std']) )
            seq = (seq-mean)/std
        all_output = []
        for c_index in range(self.n_class):


            output_after_cnn_for_lstm = input_for_cnn_before_lstm
            for i in range(0,len(self.cnn_before_lstm_layer)-1):
                output_after_cnn_for_lstm = getattr(self,f'cnn_before_lstm{i}_c{c_index}')(output_after_cnn_for_lstm)
                output_after_cnn_for_lstm = getattr(self,f'cnn_before_lstm_LR{i}_c{c_index}')(output_after_cnn_for_lstm)
       
                # output_after_cnn_for_lstm = getattr(self,f'cnn_before_lstm_pooling{i}_c{c_index}')(output_after_cnn_for_lstm)

            output_after_cnn_for_lstm = output_after_cnn_for_lstm.flatten(start_dim =2)

            ooo_batch_size = output_after_cnn_for_lstm.shape[0]
            ooo_seq_len = output_after_cnn_for_lstm.shape[1]
            output_after_cnn_for_lstm = output_after_cnn_for_lstm.reshape((ooo_seq_len,ooo_batch_size,-1))
        

            lstm_out,h = getattr(self,f'lstm_c{c_index}')(output_after_cnn_for_lstm,(getattr(self,f'h_c{c_index}'),getattr(self,f'c_c{c_index}')))
            setattr(self,f'h_c{c_index}',h)
            lstm_output = lstm_out[-1,:,-self.hidden_size:]

            cnn_output = seq

            for i in range(0,len(self.cnn_layer)-1):
            
                cnn_output = getattr(self,f'cnn{i}_c{c_index}')(cnn_output)
                cnn_output = getattr(self,f'cnn_LR{i}_c{c_index}')(cnn_output)
                cnn_output = getattr(self,f'cnn_pooling{i}_c{c_index}')(cnn_output)

            cnn_2d_output = transform2d(seq)
            cnn_2d_output = move_to_gpu(cnn_2d_output)
            for i in range(0,len(self.cnn_2d_layer)-1):
                cnn_2d_output = getattr(self,f'cnn_2d{i}_c{c_index}')(cnn_2d_output)
                cnn_2d_output = getattr(self,f'cnn_2d_LR{i}_c{c_index}')(cnn_2d_output)
        
            
        
            cnn_output=getattr(self,f'cnn_fc_c{c_index}')(cnn_output.reshape( (batch,-1) ))
            lstm_output = getattr(self,f'lstm_fc_c{c_index}')(lstm_output)
            cnn_2d_output = getattr(self,f'cnn_2d_fc_c{c_index}')(cnn_2d_output.reshape( (batch,-1) ))

            all_output.append(cnn_output)
            all_output.append(lstm_output)
            all_output.append(cnn_2d_output)
  
        # v = self.final_fc_valence_no_l2(torch.cat( (cnn_output[:,0].reshape(-1,1),lstm_output[:,0].reshape(-1,1)),1  ))
        # a = self.final_fc_arousal_no_l2(torch.cat( (cnn_output[:,1].reshape(-1,1),lstm_output[:,1].reshape(-1,1)),1))
        # pred = torch.cat((v,a),1)
        last_input_feed = torch.cat(all_output,1)
        pred = self.final_fc_no_l2(last_input_feed)

  

        return pred


class LSTM2CNN_4(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,out_size=2,number_layers=1,n_class=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_size = out_size = 1
        self.n_class = n_class
     
        # hint : input_size =  channel

        # self.cnn_layer = [input_size,100,60,50,40,30,20,20]

        # 1450 is channel according to seq_len/window_size
        self.cnn2d_layer = [1450,500,500,300,250]
        self.cnn_layer = [input_size,100,50,40,10,10]


        
        

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


  
        

            lstm_out,h = getattr(self,f'lstm_c{c_index}')(seq.view(seq_len,batch,input_size),(getattr(self,f'h_c{c_index}'),getattr(self,f'c_c{c_index}')))
            setattr(self,f'h_c{c_index}',h)
            lstm_output = lstm_out[-1,:,-self.hidden_size:]

            cnn_output = seq

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

