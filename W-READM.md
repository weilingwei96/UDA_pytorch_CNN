

- 重新下载git的项目代码
```
git clone https://github.com/weilingwei96/UDA_pytorch_CNN
```

- 需要的环境
```markdown
- python 2.7   tensorflow=1.13.1  >>  run preprocess.py
- python 3.6   torch=1.4.0        >>  run main.py
```

- 其它

    - BERT Pretrained model
    ```bash
    wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    unzip chinese_L-12_H-768_A-12.zip
    ```
    
    - 数据文件
    ```markdown
    - train.txt
    - test.txt
    - unsup.txt
    - unsup_aug.txt
    ```

- 运行步骤
    - 准备好上述环境
    - 修改代码```generate_tf/utils/raw_data_utils.py```中139行
    ```
    return ["标签1","标签2","标签3","标签4","标签5"]#["pos", "neg"] # TODO labels
    ```
    - 修改代码```load_data.py``` 193行
    ```
    labels = ('neg', 'pos') 
    ```
    - vim run_pre.sh      修改以下参数 
    ```markdown
    # input 输入数据文件  query\tlabel
    input_train=/root/wlw/UDA_pytorch/data/IMDB_raw/train.txt
    input_test=/root/wlw/UDA_pytorch/data/IMDB_raw/test.txt
    input_unsup=/root/wlw/UDA_pytorch/data/IMDB_raw/unsup.txt
    input_unsup_aug=/root/wlw/UDA_pytorch/data/IMDB_raw/unsup_aug.txt
    
    # parameters 最大长度
    max_length=25
  
    # bert pretrained model bert中文词表
    vocab=/root/wlw/UDA_pytorch/Chinese_BERT_model/vocab.txt
    
    # temp  保存proc_data path 后续还要用
    proc_save_dir_train=../data/proc_data/train_20
    proc_save_dir_test=../data/proc_data/dev
    proc_save_dir_unsup=../data/proc_data/unsup     
    ```
    - 在py27 tf113的环境下运行
    ```bash run_pre.sh ```
    - vim run_model.sh     修改以下参数
    ```markdown
    # input path     (这里的输入地址就是 temp那里的输出路径_)
    unsup_train_input=/root/wlw/UDA_pytorch_CNN/data/proc_data/unsup/bt-0.9/0/tf_examples.tfrecord.0.0
    sup_train_input=/root/wlw/UDA_pytorch_CNN/data/proc_data/train_20/tf_examples.tfrecord.0.0
    test_input=/root/wlw/UDA_pytorch_CNN/data/proc_data/dev/tf_examples.tfrecord.0.0
    
    # output path  (tfRecord转化为txt后的保存路径 记得自己先创建个子路径  例如 mkdir data/demo)
    unsup_train_output=data/demo/imdb_unsup_train.txt
    sup_train_output=data/demo/imdb_sup_train.txt
    test_output=data/demo/imdb_sup_test.txt  
    ```
    
    - 修改config/demo_uda.json文件
    ```markdown
    这里的3个路径就是上面的output path
        "sup_data_dir": "data/demo/imdb_sup_train.txt",
        "unsup_data_dir": "data/demo/imdb_unsup_train.txt",
        "eval_data_dir": "data/demo/imdb_sup_test.txt",
    
      # vocab  改成预处理时用的词表
        "vocab":"/root/wlw/UDA_pytorch_CNN/Chinese_BERT_model/vocab.txt"  
    
      # 相关参数  
    ```
    
    - 在py36 torch环境下运行 ```bash run_model.sh```
    
    
        
        
