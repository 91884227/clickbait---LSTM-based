#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[1]:


import numpy as np
import pandas as pd
import json
import torch
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.nn.functional as F
from statistics import mean 
from tqdm import tqdm
import sys


# # import self_define module

# In[2]:


from module.network_structure import LSTM_model, LSTM_model_BI, GRU_model, GRU_model_BI


# # Parameter

# In[3]:


# # FILE
# X_TRAIN_FILE = "V1_Encoding_v5_X_normal_train.json"
# y_TRAIN_FILE = "v5_y_normal_train.json"
# X_TEST_FILE = "V1_Encoding_v5_X_normal_test.json"
# y_TEST_FILE = "v5_y_normal_test.json"
# X_ANOMALY_FILE = "V1_Encoding_v5_X_anomaly_data.json"
# y_ANOMALY_FILE = "v5_Y_anomaly_data.json"
# LIMIT = 1

# # MODEL STRUCTURE
# MODEL_NAME = "LSTM_model_BI"
# INPUT_SIZE = 1 
# NUM_LAYERS = 1
# HIDDEN_SIZE = 500
# DIM_1 = 300
# DIM_2 = 50
# BS = 2
# EPOCH = 10
# LEARNING_RATE = 0.01

# # Enviroment
# DEVICE = "cuda:1"

# # Save name
# ID = "00000"


# In[ ]:


# FILE
X_TRAIN_FILE = sys.argv[1]
y_TRAIN_FILE = sys.argv[2]
X_TEST_FILE = sys.argv[3]
y_TEST_FILE = sys.argv[4]
X_ANOMALY_FILE = sys.argv[5]
y_ANOMALY_FILE = sys.argv[6]
LIMIT = int(sys.argv[7])

# MODEL STRUCTURE
MODEL_NAME = sys.argv[8]
INPUT_SIZE = int(sys.argv[9]) 
NUM_LAYERS = int(sys.argv[10])
HIDDEN_SIZE = int(sys.argv[11])
DIM_1 = int(sys.argv[12])
DIM_2 = int(sys.argv[13])
BS = int(sys.argv[14])
EPOCH = int(sys.argv[15])
LEARNING_RATE = float(sys.argv[16])

# Enviroment
DEVICE = sys.argv[17]

# Save name
ID = sys.argv[18]


# In[4]:


model_information = {"X_TRAIN_FILE" : X_TRAIN_FILE, 
                        "y_TRAIN_FILE" : y_TRAIN_FILE, 
                        "X_TEST_FILE" : X_TEST_FILE, 
                        "y_TEST_FILE" : y_TEST_FILE, 
                        "X_ANOMALY_FILE" : X_ANOMALY_FILE, 
                        "y_ANOMALY_FILE" : y_ANOMALY_FILE, 
                        "MODEL_NAME" :MODEL_NAME, 
                        "INPUT_SIZE" :INPUT_SIZE, 
                        "NUM_LAYERS" :NUM_LAYERS,
                        "HIDDEN_SIZE" :HIDDEN_SIZE,
                        "DIM_1" : DIM_1,
                        "DIM_2" : DIM_2,
                        "BS" : BS,
                        "EPOCH" : EPOCH,
                        "LEARNING_RATE" : LEARNING_RATE}


# In[5]:


with open("./model_save/info_%s.json" % ID, 'w') as outfile:
    json.dump(model_information, outfile)


# # Reproducibility 

# In[6]:


torch.manual_seed(0)
np.random.seed(0)


# # read data

# In[7]:


if( LIMIT ):
    with open('./preo_data/%s' % X_TRAIN_FILE) as json_file:
        X_train = json.load(json_file)[:1000]

    with open('./preo_data/%s' % y_TRAIN_FILE) as json_file:
        y_train = json.load(json_file)[:1000]

    with open('./preo_data/%s' % X_TEST_FILE) as json_file:
        X_test = json.load(json_file)[:1000]

    with open('./preo_data/%s' % y_TEST_FILE) as json_file:
        y_test = json.load(json_file)[:1000] 
        
    with open('./preo_data/%s' % X_ANOMALY_FILE) as json_file:
        X_anomaly = json.load(json_file)[:1000] 
        
    with open('./preo_data/%s' % y_ANOMALY_FILE) as json_file:
        y_anomaly = json.load(json_file)[:1000]  
        
else:
    with open('./preo_data/%s' % X_TRAIN_FILE) as json_file:
        X_train = json.load(json_file)

    with open('./preo_data/%s' % y_TRAIN_FILE) as json_file:
        y_train = json.load(json_file)

    with open('./preo_data/%s' % X_TEST_FILE) as json_file:
        X_test = json.load(json_file)

    with open('./preo_data/%s' % y_TEST_FILE) as json_file:
        y_test = json.load(json_file)  
        
    with open('./preo_data/%s' % X_ANOMALY_FILE) as json_file:
        X_anomaly = json.load(json_file)
        
    with open('./preo_data/%s' % y_ANOMALY_FILE) as json_file:
        y_anomaly = json.load(json_file)


# # DataLoader

# In[8]:


def create_data_loader(X_, y_, BS_):
    # x
    after_to_tensor = [torch.from_numpy(np.array(i)) for i in X_]
    after_padding = rnn_utils.pad_sequence(after_to_tensor, batch_first=True)
    after_to_numpy = after_padding.numpy()
    after_to_list = []
    
    data = torch.FloatTensor(after_padding.numpy())
    # after_resize = data.resize( data.size()[0], data.size()[1], 1 )
    after_resize = data.reshape( data.size()[0], data.size()[1], 1 )
    
    # y 1 -> [1, 0] 2-> [0, 1]
    after_list = [[1, 0] if i == 1 else [0, 1]  for i in y_]
    label = torch.FloatTensor(after_list)

    # x y to loader
    Dataset = Data.TensorDataset(after_resize, label)
    loader = torch.utils.data.DataLoader(dataset = Dataset, batch_size = BS_, shuffle = True)

    return(loader)


# In[9]:


train_loader = create_data_loader(X_train, y_train, BS_ = BS)

train_loader_BS_1 = create_data_loader(X_train, y_train, BS_ = 1)
test_loader_BS_1 = create_data_loader(X_test, y_test, BS_ = 1)
anomaly_BS_1 = create_data_loader(X_anomaly, y_anomaly, BS_ = 1)


# # model

# In[10]:


globals()[MODEL_NAME]


# In[11]:


# net = globals()[MODEL_NAME](input_size_ = 1, 
#                             num_layers_ = 1, 
#                             hidden_size_ = 500, 
#                             dim_1_ = 300, 
#                             dim_2_ = 50)
net = globals()[MODEL_NAME](input_size_ = INPUT_SIZE, 
                            num_layers_ = NUM_LAYERS, 
                            hidden_size_ = HIDDEN_SIZE, 
                            dim_1_ = DIM_1, 
                            dim_2_ = DIM_2)
net = net.to(DEVICE)


# # Loss Function

# In[12]:


loss_func = nn.BCELoss(reduction = 'sum')


# # optimize function

# In[13]:


optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)


# # Highest_value

# In[14]:


def mean_of_highest_value(loader_):
    #loader_ = train_loader_for_lambda
    with torch.no_grad():
        buf = []
        index = 0
        for x, y in loader_:
            index = index + 1
            if( index > 10000):
                break
            output = net(x.to(DEVICE))
            buf = buf + [max(output[0].data.cpu().detach().numpy())]   
        return( mean(buf) )


# # Train

# In[15]:


save_path = "./model_save/ID_%s.ptc" % ID
biggest_difference = 0

for epoch in range(EPOCH):
    train_loss = 0
    
    for x, y in tqdm(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        
        output = net(x)
        loss = loss_func(output, y)
        
        train_loss = train_loss + loss
        
        loss.backward()  
        optimizer.step()  
        
    print("\n\n===Epoch %d/%d===" % (epoch+1, EPOCH))
    print("training loss:%.2f" % (train_loss/len(train_loader)) ) 
    
    buf_1 = mean_of_highest_value(train_loader_BS_1)
    print("mean of Highest_value of normal_train %.4f" % buf_1)
    
    buf_2 = mean_of_highest_value(test_loader_BS_1)
    print("mean of Highest_value of normal_test %.4f" % buf_2)
    
    buf_3 = mean_of_highest_value(anomaly_BS_1)
    print("mean of Highest_value of anomaly %.4f" % buf_3)
    
    diff_buf2_buf3 = buf_2 - buf_3
    
    if( diff_buf2_buf3 > biggest_difference):
        print("save model at epoch %d" % (epoch+1))
        save_path = "./model_save/ID_%s_%.2f.ptc" % (ID, biggest_difference)
        torch.save(net, save_path )
        biggest_difference = diff_buf2_buf3


# In[ ]:




