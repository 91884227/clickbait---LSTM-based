#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[2]:


import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import sys
import random
from sklearn.model_selection import train_test_split


# In[11]:


if __name__ == '__main__':
    # FILENAME_ANOMALY = "replace_by_speical_tag_CNA_title_adjust.json"
    # FILENAME_NORMAL_1 = "replace_by_speical_tag_Gossip_title_20000_to_39088_adjust.json"
    # FILENAME_NORMAL_2_1 = "replace_by_speical_tag_katino_data_adjust.json"
    # FILENAME_NORMAL_2_2 = "replace_by_speical_tag_coco_title_category_1_to_121_MAX_1000_adjust.json"
    # SPILT_RATE = 0.2
    # NAME_FRONT = "v2"
    FILENAME_ANOMALY = sys.argv[1]
    FILENAME_NORMAL_1 = sys.argv[2]
    FILENAME_NORMAL_2_1 = sys.argv[3]
    FILENAME_NORMAL_2_2 = sys.argv[4]
    SPILT_RATE = float(sys.argv[5])
    NAME_FRONT = sys.argv[6]

    print("read data in...")
    with open("./preo_data/%s" % FILENAME_ANOMALY) as json_file:
        anomaly_data = json.load(json_file)

    with open("./preo_data/%s" % FILENAME_NORMAL_1) as json_file:
        normal_data_1 = json.load(json_file)

    with open("./preo_data/%s" % FILENAME_NORMAL_2_1) as json_file:
        normal_data_2_1 = json.load(json_file)

    with open("./preo_data/%s" % FILENAME_NORMAL_2_2) as json_file:
        normal_data_2_2 = json.load(json_file)    

    print("shuffle data...")
    random.shuffle(anomaly_data)
    random.shuffle(normal_data_1)
    random.shuffle(normal_data_2_1)
    random.shuffle(normal_data_2_2)

    print("deal with normal data....")
    train_2_len = min(len(normal_data_2_1), len(normal_data_2_2))*2
    train_len = min(len(normal_data_1), train_2_len)

    X_normal_data = np.array(normal_data_1[:2*(train_len//2)] + normal_data_2_1[:train_len//2] + normal_data_2_1[:train_len//2])
    X_normal_data = normal_data_1[:2*(train_len//2)] + normal_data_2_1[:train_len//2] + normal_data_2_1[:train_len//2]
    Y_normal_data = list(np.repeat((1, 2), len(X_normal_data)/2))

    print("split normal data...")
    X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(X_normal_data, 
                                                                                    Y_normal_data, 
                                                                                    test_size = SPILT_RATE)

    # for json save
    y_normal_train = [int(i) for i in y_normal_train]
    y_normal_test = [int(i) for i in y_normal_test]

    print("save normal spilt data...")
    with open("./preo_data/%s_X_normal_train.json" % NAME_FRONT, 'w') as outfile:
        json.dump(X_normal_train, outfile)

    with open("./preo_data/%s_X_normal_test.json" % NAME_FRONT, 'w') as outfile:
        json.dump(X_normal_test, outfile)

    with open("./preo_data/%s_y_normal_train.json" % NAME_FRONT, 'w') as outfile:
        json.dump(y_normal_train, outfile)

    with open("./preo_data/%s_y_normal_test.json" % NAME_FRONT, 'w') as outfile:
        json.dump(y_normal_test, outfile)

    print("anomaly data proprocessing..")
    X_anomaly_data = anomaly_data
    Y_anomaly_data = list(np.repeat(0, len(X_anomaly_data)))

    # for json save
    Y_anomaly_data = [int(i) for i in Y_anomaly_data]

    with open("./preo_data/%s_X_anomaly_data.json" % NAME_FRONT, 'w') as outfile:
        json.dump(X_anomaly_data, outfile)

    with open("./preo_data/%s_Y_anomaly_data.json" % NAME_FRONT, 'w') as outfile:
        json.dump(Y_anomaly_data, outfile)

