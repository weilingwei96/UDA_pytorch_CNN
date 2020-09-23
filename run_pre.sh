#!/usr/bin/env bash

#conda activate py27-tf113
# input
input_train=../data/IMDB_raw/train.txt
input_test=../data/IMDB_raw/test.txt
input_unsup=../data/IMDB_raw/unsup.txt
input_unsup_aug=../data/IMDB_raw/unsup_aug.txt

# parameters
max_length=25
# bert pretrained model
vocab=../Chinese_BERT_model/vocab.txt

# temp  save proc_data path
proc_save_dir_train=../data/proc_data/train_20
proc_save_dir_test=../data/proc_data/dev
proc_save_dir_unsup=../data/proc_data/unsup \



# preprocess.py
cd generate_tf
mkdir ../data/
mkdir ../data/proc_data/

# train
python -u preprocess.py  \
        --raw_data_dir=${input_train}  \
        --output_base_dir=${proc_save_dir_train}  \
        --data_type=sup   \
        --sub_set=train   \
        --sup_size=-1   \
        --vocab_file=${vocab} \
        --max_seq_length=${max_length}

# test
python -u preprocess.py  \
        --raw_data_dir=${input_test}  \
        --output_base_dir=${proc_save_dir_test}  \
        --data_type=sup   \
        --sub_set=dev  \
        --vocab_file=${vocab} \
        --max_seq_length=${max_length}


# train-unsup
python -u preprocess.py  \
        --raw_data_dir=${input_unsup}  \
        --aug_raw_data_dir=${input_unsup_aug}  \
        --output_base_dir=${proc_save_dir_unsup} \
        --data_type=unsup   \
        --sub_set=unsup_in \
        --aug_ops=bt-0.9  \
        --aug_copy_num=0  \
        --vocab_file=${vocab} \
        --max_seq_length=${max_length}

