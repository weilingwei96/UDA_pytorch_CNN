#!/usr/bin/env bash


## transfer tfrecorder

# input path
unsup_train_input=/root/wlw/UDA_pytorch/data/proc_data/unsup/bt-0.9/0/tf_examples.tfrecord.0.0
sup_train_input=/root/wlw/UDA_pytorch/data/proc_data/train_20/tf_examples.tfrecord.0.0
test_input=/root/wlw/UDA_pytorch/data/proc_data/dev/tf_examples.tfrecord.0.0

# output path
unsup_train_output=/root/wlw/UDA_pytorch/demo/imdb_unsup_train.txt
sup_train_output=/root/wlw/UDA_pytorch/demo/imdb_sup_train.txt
test_output=/root/wlw/UDA_pytorch/demo/imdb_sup_test.txt

echo 'transfer...'
echo $unsup_train_input+'>'+$unsup_train_output
echo $sup_train_input+'>'+$sup_train_output
echo $test_input+'>'+$test_output

python -u tfRecorder_to_txt.py \
    --unsup_train_input_data=${unsup_train_input} \
    --unsup_train_output_data=${unsup_train_output} \
    --sup_train_input_data=${sup_train_input} \
    --sup_train_output_data=${sup_train_output} \
    --sup_test_input_data=${test_input} \
    --sup_test_output_data=${test_output}

echo '\n\n\n\n\n'
echo 'UDA - pytorch running...'
cat config/demo_uda.json
# change uda_config path 
python -u main.py \
      --uda_config='config/demo_uda.json' \
      --bert_base_config='config/chinese_bert_base.json'

