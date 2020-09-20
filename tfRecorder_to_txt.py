import numpy as np
import os
import tensorflow as tf
import pandas as pd
# sup_train/sup_test
# input_ids=input_ids,
#             input_mask=input_mask,
#             input_type_ids=input_type_ids,
#             label_id=label_id))

# unsup
# def get_dict_features(self):
#     return {
#         "ori_input_ids": _create_int_feature(self.ori_input_ids),
#         "ori_input_mask": _create_int_feature(self.ori_input_mask),
#         "ori_input_type_ids": _create_int_feature(self.ori_input_type_ids),
#         "aug_input_ids": _create_int_feature(self.aug_input_ids),
#         "aug_input_mask": _create_int_feature(self.aug_input_mask),
#         "aug_input_type_ids": _create_int_feature(self.aug_input_type_ids),
#     }




def read_data(path, type, outpath ):
    datas = []
    if type=='unsup':
        name = ['ori_input_ids', 'ori_input_mask', 'ori_input_type_ids', 'aug_input_ids', 'aug_input_mask',
            'aug_input_type_ids']
    elif type=='train' or type=='test':
        name = ['input_ids','input_mask', 'input_type_ids', 'label_ids']

    # return
    # path = os.path.join('data/proc_data/IMDB/unsup', 'bt-0.9/0/tf_examples.tfrecord.0.0')
    iter = tf.python_io.tf_record_iterator(path)
    num = 0
    for serialized_example in iter:
        data = []
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        if type=='unsup':

            ori_input_ids = example.features.feature['ori_input_ids'].int64_list.value
            ori_input_mask = example.features.feature['ori_input_mask'].int64_list.value
            ori_input_type_ids = example.features.feature['ori_input_type_ids'].int64_list.value
            aug_input_ids = example.features.feature['aug_input_ids'].int64_list.value
            aug_input_mask = example.features.feature['aug_input_mask'].int64_list.value
            aug_input_type_ids = example.features.feature['aug_input_type_ids'].int64_list.value

            ori_input_ids = list(np.array(ori_input_ids).astype(np.int64))
            ori_input_mask = list(np.array(ori_input_mask).astype(np.int64))
            ori_input_type_ids = list(np.array(ori_input_type_ids).astype(np.int64))
            aug_input_ids = list(np.array(aug_input_ids).astype(np.int64))
            aug_input_mask = list(np.array(aug_input_mask).astype(np.int64))
            aug_input_type_ids = list(np.array(aug_input_type_ids).astype(np.int64))


            data.append(ori_input_ids)
            data.append(ori_input_mask)
            data.append(ori_input_type_ids)
            data.append(aug_input_ids)
            data.append(aug_input_mask)
            data.append(aug_input_type_ids)
        else:
            input_ids = example.features.feature['input_ids'].int64_list.value
            input_mask = example.features.feature['input_mask'].int64_list.value
            input_type_ids = example.features.feature['input_type_ids'].int64_list.value
            label_ids = example.features.feature['label_ids'].int64_list.value


            input_ids = list(np.array(input_ids).astype(np.int64))
            input_mask = list(np.array(input_mask).astype(np.int64))
            input_type_ids = list(np.array(input_type_ids).astype(np.int64))
            label_ids = int(np.array(label_ids).astype(np.int64))


            data.append(input_ids)
            data.append(input_mask)
            data.append(input_type_ids)
            data.append(label_ids)
        datas.append(data)
        num+=1
        if num%10==0:
            print("Load {}".format(num ))
            break
    df = pd.DataFrame(datas,columns=name)

    df.to_csv(outpath,sep='\t',index=0)
    print("Save ok.", outpath)

path = os.path.join('data/proc_data/IMDB/unsup', 'bt-0.9/0/tf_examples.tfrecord.0.0')
outpath = "/root/wlw/UDA_pytorch/demo/imdb_unsup_train.txt"
read_data(path,'unsup', outpath)

path = '/root/wlw/uda/data/proc_data/IMDB/train_20/tf_examples.tfrecord.0.0'
outpath = "/root/wlw/UDA_pytorch/demo/imdb_sup_train.txt"
read_data(path,'train', outpath)

path = '/root/wlw/uda/data/proc_data/IMDB/dev/tf_examples.tfrecord.0.0'
outpath = "/root/wlw/UDA_pytorch/demo/imdb_sup_test.txt"
read_data(path,'test', outpath)