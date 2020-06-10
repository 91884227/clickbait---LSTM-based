import json
import sys
from tqdm import tqdm

from module.replace_by_tag_class import class_replace_by_tag

TAG_LIST = sys.argv[1]
POS_FILE = sys.argv[2]
WS_FILE = sys.argv[3]
ID = sys.argv[4]

replace_tag_func = class_replace_by_tag(TAG_LIST)

if __name__ == '__main__':
    temp = "./preprocess_data/%s"

    with open( temp % POS_FILE) as json_file:
        pos_data = json.load(json_file)

    with open( temp % WS_FILE) as json_file:
        ws_data = json.load(json_file)

    save_data = [replace_tag_func(i, j) for i, j in tqdm(zip( pos_data, ws_data)) ]

    with open(temp % ID, 'w') as outfile:
        json.dump(save_data, outfile)