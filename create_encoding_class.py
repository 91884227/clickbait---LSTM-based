#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[123]:


from sklearn import preprocessing
import json
import pandas as pd
import numpy as np
from itertools import chain
from collections import Counter
import pickle
import sys


# # Find common word and save

# In[124]:


def find_common_word(list_, most_common_number_):
    c = Counter(list_)
    c_1 = c.most_common(most_common_number_)
    buf = common_word_list = [i[0] for i in c_1]
    return(buf)


# #  create Encoding function

# In[125]:


def create_encoding_function(list_):
    le = preprocessing.LabelEncoder()
    le.fit(list_ + ["<UNK>"])
    return(le)

# le.transform(['保持','閱讀','以及','的'])
# list(le.classes_)
# le.d


# In[134]:


if __name__ == '__main__':
#     FILENAME = "v5_X_normal_train.json"
#     MOST_COMMON_WORD = 10000
#     SAVE_PRE_NAME = "v1_Encoding_function.pkl"
    FILENAME = sys.argv[1]
    MOST_COMMON_WORD = int(sys.argv[2])
    SAVE_PRE_NAME = sys.argv[3]

    print("read data in...")
    with open('./preo_data/%s' % FILENAME) as json_file:
        data = json.load(json_file)

    data = list(chain.from_iterable(data))

    print("find most common word...")
    most_common_word = find_common_word(data, MOST_COMMON_WORD)

    print("create encoding function...")
    le = create_encoding_function(most_common_word) # le = preprocessing.LabelEncoder()

    print("save class...")
    with open('./preo_data/%s_Encoding_function.pkl' % SAVE_PRE_NAME, 'wb') as output:
        pickle.dump(le, output, pickle.HIGHEST_PROTOCOL)

