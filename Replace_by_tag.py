#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[126]:


import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import sys


# In[127]:


def replace_str(pos, ws, tag = ["Nb", "Neu", "Nc", "Nd", "Nf"]):
    buf = ws.copy()
    #print(buf)
    #print(pos)
    for index, i_pos in enumerate(pos):
        if( i_pos in tag):
            buf[index] = i_pos
    return(buf)


# In[132]:


if __name__ == '__main__':
    FILENAME_POS = sys.argv[1]
    FILENAME_ws = sys.argv[2]
    LIMIT = int(sys.argv[3])
#     FILENAME_POS = "Gossip_title_20000_to_39088_adjust_POS.json"
#     FILENAME_ws = "Gossip_title_20000_to_39088_adjust_ws.json"

    FILENAME = FILENAME_ws[:-8]
    
    print("read data")
    with open("./ori_data/%s" % FILENAME_POS) as json_file:
        data_POS = json.load(json_file)

    with open("./ori_data/%s" % FILENAME_ws) as json_file:
        data_ws = json.load(json_file)

    print("start replace...")
    if(LIMIT ):
        buf = [replace_str(data_POS[i], data_ws[i]) for i in tqdm(range(100))]
    else:
        buf = [replace_str(data_POS[i], data_ws[i]) for i in tqdm(range(len(data_ws)))]

    output_name = "./preo_data/replace_by_speical_tag_%s.json" % FILENAME

    try:
        with open(output_name, 'w') as outfile:
            json.dump(buf, outfile)
        print("success to create %s" % output_name)
    except:
        print("fail")

