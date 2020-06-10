import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import sys

class class_replace_by_tag:
    def __init__(self, tag_file_):
        file_path = "./module/%s" % tag_file_
        f = open(file_path, 'r+')
        self.tag = f.read().splitlines()
        f.close()
    def __call__(self, ws_, pos_):
        buf = ws_.copy()
        for index, i_pos in enumerate(pos_):
            if( i_pos in self.tag):
                buf[index] = i_pos
        return(buf)        

# REPLACE_TAG_FILE = "replace_tag.txt"
# test = class_replace_by_tag(REPLACE_TAG_FILE )
# pos = ['武漢', '肺炎', '23日', '零', '確診', ' ', '連', '41', '天', '無', '本土', '病例']
# ws =['Nc', 'Na', 'Nd', 'Neu','VD','WHITESPACE','Cbb','Neu','Nf','VJ','Nc','Na']
# test(pos, ws)
