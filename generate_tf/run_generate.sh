#!/usr/bin/env bash
python preprocess.py  \
        --raw_data_dir=../data/IMDB_raw/unsup.txt  \
        --aug_raw_data_dir=../data/IMDB_raw/unsup_aug.txt  \
        --output_base_dir=../data/proc_data/unsup \
        --back_translation_dir=back_translation/imdb_back_trans  \
        --data_type=unsup   \
        --sub_set=unsup_in \
        --aug_ops=bt-0.9  \
        --aug_copy_num=0  \
        --vocab_file=../Chinese_BERT_model/vocab.txt \
        --max_seq_length=25


