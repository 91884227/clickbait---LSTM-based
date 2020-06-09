import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import json
import sys

from module.sentence_segmentation_class import class_sentence_segmentation
postag = class_sentence_segmentation(CUDA_VISIBLE_DEVICES = "0", GPU_MEMORY_FRACTION = 0.7)

# # Parameter
# DATA_NAME = "CNA_title_adjust.npy"
# ID = "CNA"
# ifMULTIPROCESS = 0
DATA_NAME = sys.argv[1]
ID = sys.argv[2]

if __name__ == '__main__':
    # read data in
    path = "./raw_data/%s" % DATA_NAME
    data = np.load(path)

    # start processing
    result = [postag(i) for i in tqdm(data)]
    pos = [i[0] for i in result]
    ws = [i[1] for i in result]

    # save data
    temp = "./preprocess_data/%s_%s.json"
    with open(temp % (ID, "pos"), 'w') as outfile:
        json.dump(pos, outfile)
    with open(temp % (ID, "ws"), 'w') as outfile:
        json.dump(ws, outfile)