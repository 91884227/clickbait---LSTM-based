# clickbait---LSTM-based

## train_model.py
### usage
```
python train_model.py X_TRAIN_FILE y_TRAIN_FILE X_TEST_FILE y_TEST_FILE X_ANOMALY_FILE y_ANOMALY_FILE LIMIT MODEL_NAME INPUT_SIZE NUM_LAYERS HIDDEN_SIZE DIM_1 DIM_2 BS EPOCH LEARNING_RATE DEVICE ID
```
| Parameter | meaning | e.g. |
| -------- | -------- | -------- |
| X_TRAIN_FILE | $*$ 須放在 `./preo_data/`下 | "V1_Encoding_v5_X_normal_train.json" |
| y_TRAIN_FILE | $*$ 須放在 `./preo_data/`下 | "v5_y_normal_train.json" |
| X_TEST_FILE | $*$ 須放在 `./preo_data/`下 | "V1_Encoding_v5_X_normal_test.json" |
| y_TEST_FILE |  $*$ 須放在 `./preo_data/`下| "v5_y_normal_test.json" |
| X_ANOMALY_FILE | $*$ 須放在 `./preo_data/`下 | "V1_Encoding_v5_X_anomaly_data.json" |
| y_ANOMALY_FILE | $*$ 須放在 `./preo_data/`下 | "v5_Y_anomaly_data.json" |
| LIMIT | 0: 讀全部檔案 </br> 1: 讀前100筆檔案 | 1 |
| MODEL_NAME | 可選擇</br> LSTM_model</br>LSTM_model_BI</br> GRU_model</br>GRU_model_BI| "LSTM_model_BI" |
| INPUT_SIZE | network parameter | 1 |
| NUM_LAYERS | network parameter | 1 |
| HIDDEN_SIZE | network parameter | 500 |
| DIM_1 | network parameter | 300 |
| DIM_2 | network parameter | 50 |
| BS | Batch_size | 2 |
| EPOCH |  | 10 |
| LEARNING_RATE |  | 0.01 |
| DEVICE | GPU名稱 | "cuda:1" |
| ID | meaning | "00000" |

則可訓練以`MODEL_NAME`為架構的模型
將模型命名為 `ID_ID.ptc`儲存在`./model_save/`下
模型的相關資訊為 `info_ID.json`儲存在`./model_save/`下



### example
```
python train_model.py "V1_Encoding_v5_X_normal_train.json" "v5_y_normal_train.json" "V1_Encoding_v5_X_normal_test.json" "v5_y_normal_test.json" "V1_Encoding_v5_X_anomaly_data.json" "v5_Y_anomaly_data.json" 1 "LSTM_model_BI" 1 1 500 300 50 2 10 0.01 "cuda:1" "00000"
```
則可訓練以`LSTM_model_BI`為架構的模型
將模型命名為 `ID_00000.ptc`儲存在`./model_save/`下
模型的相關資訊為 `info_00000.json`儲存在`./model_save/`下

## encoding_data.py
依據 `CLASSNAME` 將資料編碼 
* Rmk 沒出現用 \<UNK\> 來表示
### usage
```
python encoding_data.py FILENAME CLASSNAME MAX_LEN LIMIT OUTPUT_PRE_NAME 
```
| Parameter | meaning | e.g. |
| -------- | -------- | -------- |
| FILENAME     |   等待編碼的檔案名稱 </br> $*$須放在 `./preo_data/`下   | "v5_X_normal_train.json" |
| CLASSNAME     |   function的檔案名稱 </br> $*$須放在 `./preo_data/`下   | "v5_Encoding_function.pkl" |
| MAX_LEN | 超過MAX_LEN 會被CUT掉| 20 |
| LIMIT | 0: 處理全部資料</br>1: 處理前100筆資料 | 0 |
| OUTPUT_PRE_NAME | 輸出檔案時的前綴字 | "V1"|

即可產生`OUTPUT_PRE_NAME_Encoding_FILENAME.json`的檔案在 `./preo_data/`下
### example
```
python encoding_data.py "v5_X_normal_train.json" "v5_Encoding_function.pkl" 20 0 "V1" 
```
即可產生`V1_Encoding_v5_X_normal_train.json`的檔案在 `./preo_data/`下

## create_encoding_class.py
### usage
```
python create_encoding_class.py FILENAME MOST_COMMON_WORD SAVE_PRE_NAME 
```
| Parameter | meaning | e.g. |
| -------- | -------- | -------- |
| FILENAME     |   等待編碼的檔案名稱 </br> $*$須放在 `./preo_data/`下   | "v5_X_normal_train.json" |
| MOST_COMMON_WORD     | 取頻率前MOST_COMMON_WORD的字</br> 其他視為Unknown    | 10000 |
| SAVE_PRE_NAME      |  輸出檔案時的前綴字     | "v1" |
即可產生`SAVE_PRE_NAME_Encoding_function.pkl`的檔案在 `./preo_data/`下
### example
```
python create_encoding_class.py "v5_X_normal_train.json" 10000 "v1"
```
即可產生`v1_Encoding_function.pkl`的檔案在 `./preo_data/`下


## split_data_set.py
將 raw data 分割成 training testing 以及 normal
![](https://i.imgur.com/q3C6mrA.png)
### usage
```
python split_data_set.py FILENAME_ANOMALY FILENAME_NORMAL_1 FILENAME_NORMAL_2_1 FILENAME_NORMAL_2_2 SPILT_RATE NAME_FRONT
```


| Parameter | meaning | e.g. |
| -------- | -------- | -------- |
| FILENAME_ANOMALY     | 檔案名稱</br> $*$ 須放在 `./preo_data/`下     | "replace_by_speical_tag_CNA_title_adjust.json"     |
| FILENAME_NORMAL_1     | 檔案名稱</br> $*$ 須放在 `./preo_data/`下    | "replace_by_speical_tag_Gossip_title_20000_to_39088_adjust.json"     |
| FILENAME_NORMAL_2_1     | 檔案名稱 </br> $*$ 須放在 `./preo_data/`下    | "replace_by_speical_tag_katino_data_adjust.json"     |
| FILENAME_NORMAL_2_2     | 檔案名稱 </br> $*$ 須放在 `./preo_data/`下    | "replace_by_speical_tag_coco_title_category_1_to_121_MAX_1000_adjust.json"     |
| SPILT_RATE     | TESTING 的資料比例    | 0.2    |
| NAME_FRONT    | 輸出檔案時的前綴字     | "V1"     |

即可輸出


| Filename | 
| -------- |
| NAME_FRONT_X_anomaly_data.json     |
| NAME_FRONT_X_normal_test.json     |
| NAME_FRONT_X_normal_train.json     | 
| NAME_FRONT_Y_anomaly_data.json     | 
| NAME_FRONT_Y_normal_test.json      |
| NAME_FRONT_Y_normal_test.json n     | 


### example
```
python split_data_set.py "replace_by_speical_tag_CNA_title_adjust.json" "replace_by_speical_tag_Gossip_title_20000_to_39088_adjust.json" "replace_by_speical_tag_katino_data_adjust.json" "replace_by_speical_tag_coco_title_category_1_to_121_MAX_1000_adjust.json" 0.2 "v1"
```

即可輸出
| Filename | 
| -------- |
| V1_X_anomaly_data.json     |
| V1_X_normal_test.json     |
| V1_X_normal_train.json     | 
| V1_Y_anomaly_data.json     | 
| V1_Y_normal_test.json      |
| V1_Y_normal_test.json n     | 



## Replace_by_tag.py
### What this program do 
將 [POS Tags](https://github.com/ckiplab/ckiptagger/wiki/POS-Tags) 為 ["Nb", "Neu", "Nc", "Nd", "Nf"] 做替代
e.g. 

ws: ['台灣', '司法', '天秤', '是', '不', '是', '真的', '不', '一樣', '的', '卦', '?', '!']

pos: ['Nc','Na', 'Na', 'SHI', 'D', 'SHI', 'D', 'D', 'VH','DE','Na','QUESTIONCATEGORY', 'EXCLAMATIONCATEGORY']

取代完後:['Nc', '司法', '天秤', '是', '不', '是', '真的', '不', '一樣', '的', '卦', '?', '!']
 



### usage
```
python Replace_by_tag.py FILENAME_POS FILENAME_ws LIMIT
```
* Rmk

| Parameter | meaning | e.g. |
| -------- | -------- | -------- |
|FILENAME_POS  | pos的檔案名稱 </br> $*$ 須放在 `./ori_data/`下     | "Gossip_title_20000_to_39088_adjust_POS.json"     |
|FILENAME_ws  | ws的檔案名稱  </br> $*$ 須放在 `./ori_data/`下     | "Gossip_title_20000_to_39088_adjust_ws.json"    |
|LIMIT| 0: 處理全部資料 </br>1: 處理前100筆    | 1  |

跑完即生出 `replace_by_speical_tag_FILENAME.json` 在 `./preo_data/` 下

### example
```
python Replace_by_tag.py "Gossip_title_20000_to_39088_adjust_POS.json" "Gossip_title_20000_to_39088_adjust_ws.json" 1
```
跑完即生出 `replace_by_speical_tag_Gossip_title_20000_to_39088_adjust.json` 在 `./preo_data/` 下





