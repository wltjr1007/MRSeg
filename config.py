import argparse


parser = argparse.ArgumentParser()
parser.set_defaults(train=True)
parser.set_defaults(deploy=False)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.add_argument('--deploy', dest='deploy', action='store_true')
parser.add_argument('--mode', type=int, default=0)
parser.add_argument('--GPU', type=int, default=-1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--tst_ckpt', type=int, default=-1)
parser.add_argument('--resume', type=int, default=0)
ARGS = parser.parse_args()

mode_dict = {0: "segment", 1: "stage"}

is_train= ARGS.train
deploy= ARGS.deploy
mode = ARGS.mode
GPU = ARGS.GPU
seed = ARGS.seed
tst_ckpt = ARGS.tst_ckpt
resume = ARGS.resume

import os
from datetime import datetime
import GPUtil
import numpy as np
import tensorflow as tf

if seed!=-1:
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)
if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU
os.environ["CUDA_VISIBLE_DEVICES"] = devices
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.get_logger().setLevel("ERROR")

root_path = "/home/jsyoon/data/project/"
project_path = root_path+"deeplab3p_mr/"
raw_data_path = root_path+"NPY_data/"
result_path = "/home/jsyoon/data_ssd/summ/mr_seg/%s/%s/"%(mode_dict[mode], datetime.now().strftime("%m%d_%H%M%S%f")[:-3])


epoch_cnt = 200
iter_cnt = 100000
batch_size =4
stg_batch_size = 5

seg_D = 3
W=512
H=512

resize_W=128
resize_H = 128
resize_D=96
stg_D = resize_D
in_feat = 2

rank=3
weight_decay=0.00005
weight_const = 2.5

data_type= "float32"