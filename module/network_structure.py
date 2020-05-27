#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[7]:


import torch
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.nn.functional as F


# In[6]:


class LSTM_model(nn.Module):
    def __init__(self, input_size_ = 1, num_layers_ = 1, hidden_size_ = 2, dim_1_ = 300, dim_2_ = 50):
        super(LSTM_model, self).__init__()
        self.input_size = input_size_
        self.hidden_size = hidden_size_
        self.num_layers = num_layers_
        self.dim_1 = dim_1_
        self.dim_2 = dim_2_
#         self.input_size = 1
#         self.hidden_size = 8
#         self.num_layers = 4
#         self.dim_1 = 5
#         self.dim_2 = 2
        
        self.RNN_base = nn.LSTM(input_size = self.input_size, 
                            hidden_size = self.hidden_size, 
                            num_layers = self.num_layers, 
                            #dropout = 0.5, 
                            bidirectional = False,
                            batch_first = True)
        self.fc1 = nn.Linear(self.hidden_size, self.dim_1)
        self.fc2 = nn.Linear(self.dim_1, self.dim_2)
        self.fc3 = nn.Linear(self.dim_2, 2)
    
    def forward(self, x):
        r_out, (hn, cn) = self.RNN_base(x)
        x = F.relu(r_out[:, -1, :])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim = 1)
        return(x)


# In[8]:


class LSTM_model_BI(nn.Module):
    def __init__(self, input_size_ = 1, num_layers_ = 1, hidden_size_ = 2, dim_1_ = 300, dim_2_ = 50):
        super(LSTM_model_BI, self).__init__()
        self.input_size = input_size_
        self.hidden_size = hidden_size_
        self.num_layers = num_layers_
        self.dim_1 = dim_1_
        self.dim_2 = dim_2_
#         self.input_size = 1
#         self.hidden_size = 8
#         self.num_layers = 4
#         self.dim_1 = 5
#         self.dim_2 = 2
        
        self.RNN_base = nn.LSTM(input_size = self.input_size, 
                            hidden_size = self.hidden_size, 
                            num_layers = self.num_layers, 
                            #dropout = 0.5, 
                            bidirectional = True,
                            batch_first = True)
        self.fc1 = nn.Linear(self.hidden_size*2, self.dim_1)
        self.fc2 = nn.Linear(self.dim_1, self.dim_2)
        self.fc3 = nn.Linear(self.dim_2, 2)
    
    def forward(self, x):
        r_out, (hn, cn) = self.RNN_base(x)
        x = F.relu(r_out[:, -1, :])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim = 1)
        return(x)


# In[ ]:
class GRU_model(nn.Module):
    def __init__(self, input_size_ = 1, num_layers_ = 1, hidden_size_ = 2, dim_1_ = 300, dim_2_ = 50):
        super(GRU_model, self).__init__()
        self.input_size = input_size_
        self.hidden_size = hidden_size_
        self.num_layers = num_layers_
        self.dim_1 = dim_1_
        self.dim_2 = dim_2_
#         self.input_size = 1
#         self.hidden_size = 8
#         self.num_layers = 4
#         self.dim_1 = 5
#         self.dim_2 = 2
        
        self.RNN_base = nn.LSTM(input_size = self.input_size, 
                            hidden_size = self.hidden_size, 
                            num_layers = self.num_layers, 
                            #dropout = 0.5, 
                            bidirectional = False,
                            batch_first = True)
        self.fc1 = nn.Linear(self.hidden_size, self.dim_1)
        self.fc2 = nn.Linear(self.dim_1, self.dim_2)
        self.fc3 = nn.Linear(self.dim_2, 2)
    
    def forward(self, x):
        r_out, hn = self.RNN_base(x)
        x = F.relu(r_out[:, -1, :])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim = 1)
        return(x)

class GRU_model_BI(nn.Module):
    def __init__(self, input_size_ = 1, num_layers_ = 1, hidden_size_ = 2, dim_1_ = 300, dim_2_ = 50):
        super(GRU_model_BI, self).__init__()
        self.input_size = input_size_
        self.hidden_size = hidden_size_
        self.num_layers = num_layers_
        self.dim_1 = dim_1_
        self.dim_2 = dim_2_
#         self.input_size = 1
#         self.hidden_size = 8
#         self.num_layers = 4
#         self.dim_1 = 5
#         self.dim_2 = 2
        
        self.RNN_base = nn.LSTM(input_size = self.input_size, 
                            hidden_size = self.hidden_size, 
                            num_layers = self.num_layers, 
                            #dropout = 0.5, 
                            bidirectional = True,
                            batch_first = True)
        self.fc1 = nn.Linear(self.hidden_size*2, self.dim_1)
        self.fc2 = nn.Linear(self.dim_1, self.dim_2)
        self.fc3 = nn.Linear(self.dim_2, 2)
    
    def forward(self, x):
        r_out, hn = self.RNN_base(x)
        x = F.relu(r_out[:, -1, :])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim = 1)
        return(x)



