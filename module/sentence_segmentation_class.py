#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[1]:


import pandas as pd
import json
import numpy as np
import itertools
import collections
import tensorflow as tf
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import os
from datetime import date
from IPython.display import clear_output
clear_output()
from tqdm import tqdm
import sys
import pickle


class class_sentence_segmentation:
    def __init__(self, GPU_MEMORY_FRACTION, CUDA_VISIBLE_DEVICES):
        print("set GPU stat...")
        cfg = tf.ConfigProto()
        cfg.gpu_options.per_process_gpu_memory_fraction =  GPU_MEMORY_FRACTION ###設定gpu使用量
        session = tf.Session(config = cfg)
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES ###設定gpu編號
        
        print("prepare ws pos ner")
        path = "./module/data"
        self.ws = WS(path, disable_cuda = False)
        self.pos = POS(path, disable_cuda = False)
        ner = NER(path, disable_cuda = False)
        clear_output() 
    
    def __call__(self, str_):
        word_sentence_list = self.ws(
            [str_],
            sentence_segmentation = True, # To consider delimiters
            # segment_delimiter_set = {",", "。", ":", "?", "!", ";"}), # This is the defualt set of delimiters
            # recommend_dictionary = dictionary1, # words in this dictionary are encouraged
            # coerce_dictionary = dictionary2, # words in this dictionary are forced
        )   
        pos_sentence_list = self.pos(word_sentence_list)
        return( (word_sentence_list[0], pos_sentence_list[0]) )

#test = class_sentence_segmentation(CUDA_VISIBLE_DEVICES = "0", GPU_MEMORY_FRACTION = 0.7)
#test("北約硬起來！ 秘書長籲全球共同對抗中國霸凌")

