#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[41]:


import pandas as pd
import numpy as np
import pickle
import json
from tqdm import tqdm
import sys


# In[42]:


if __name__ == '__main__':
    # FILE_PATH = "v5_X_normal_train.json"
    # CLASS_PATH = "v5_Encoding_function.pkl"
    # MAX_LEN = 20
    # LIMIT = 1
    # OUTPUT_PRE_NAME = "V1"
    
    FILENAME = sys.argv[1]
    CLASSNAME = sys.argv[2]
    MAX_LEN = int(sys.argv[3])
    LIMIT = int(sys.argv[4])
    OUTPUT_PRE_NAME = sys.argv[5]

    print("read file in..")
    with open("./preo_data/%s" % FILENAME) as json_file:
        data = json.load(json_file)

    with open("./preo_data/%s" % CLASSNAME, 'rb') as input:
        le = pickle.load(input)

    if( LIMIT ):
        data = data[:100]

    print("processing data...")
    command_word_list = list(le.classes_)
    after_replace = [ [i if i in command_word_list else "<UNK>" for i in str_] for str_ in tqdm(data)]
    after_len_selection = [i if len(i)<MAX_LEN else i[:MAX_LEN] for i in after_replace]
    after_encode = [le.transform(i) for i in tqdm(after_len_selection)] 
    
    # for json saveing
    after_int = [ list(map(int,  i)) for i in after_encode]  

    print("save data...")
    output_name = "%s_Encoding_%s.json" % (OUTPUT_PRE_NAME, FILENAME[:-5])
    with open('./preo_data/%s' % output_name , 'w') as outfile:
        json.dump(after_int, outfile)

