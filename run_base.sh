#!/usr/bin/env bash

## transfer tfrecorder

#python -u tfRecorder_to_txt.py \
#    --unsup_train_input_data=/root/wlw/uda/data/proc_data/IMDB/unsup/bt-0.9/0/tf_examples.tfrecord.0.0 \
#    --unsup_train_output_data=/root/wlw/UDA_pytorch/demo/imdb_unsup_train.txt \
#    --sup_train_input_data=/root/wlw/uda/data/proc_data/IMDB/train_20/tf_examples.tfrecord.0.0 \
#    --sup_train_output_data=/root/wlw/UDA_pytorch/demo/imdb_sup_train.txt \
#    --sup_test_input_data=/root/wlw/uda/data/proc_data/IMDB/dev/tf_examples.tfrecord.0.0 \
#    --sup_test_output_data=/root/wlw/UDA_pytorch/demo/imdb_sup_test.txt

echo config/non-uda.json
echo 'UDA - pytorch running...'
python -u main.py \
      --uda_config='config/non-uda.json' \
      --bert_base_config='config/bert_base.json'


