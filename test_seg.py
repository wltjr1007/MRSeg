try:
    import tensorflow as tf
except:
    raise ImportError("Tensorflow is required")
import re
if int(tf.__version__.split(".")[0])!=2:
    raise ImportError("Your Tensorflow version is %s. Tensorflow version 2 is required (2.0.0 is recommended)."%tf.__version__)

from glob import glob
import numpy as np
import os
from PIL import Image
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cur_dtype = "float32"
tf.keras.backend.set_floatx(cur_dtype)
from model import Deeplabv3

import GPUtil
os.environ["CUDA_VISIBLE_DEVICES"]="%d" % GPUtil.getFirstAvailable(order="memory")[0]
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])

import sys
try:
    ckpt = sys.argv[1]
except:
    ckpt = "./model.pickle"
    #"/mnt/SSD_BIG1/jsyoon/project/deeplab3p/summ/segment/0902_230904918/model/119/"
try:
    in_dir = sys.argv[2]
except:
    in_dir = "./dcm_data/"

if not os.path.exists(ckpt):
    raise FileNotFoundError("Model file is not found")
if not os.path.exists(in_dir):
    raise FileNotFoundError("Input directory is not found")

def load_data():
    def load_dcm_data(fn):
        try:
            import pydicom
        except:
            raise ImportError("To read DCM files, pydicom is required.")
        temp_dat = []
        temp_idx = []
        for cnt, f in enumerate(fn):
            dat = pydicom.dcmread(f, force=True)
            dat.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"
            try:
                temp_idx += [int(dat.InstanceNumber)]
            except:
                try:
                    temp_idx += [int(dat.AcquisitionNumber)]
                except:
                    raise Exception("Slice number metaData not found")
            temp_dat += [dat.pixel_array]
        temp_idx = np.array(temp_idx)
        new_dat = np.zeros(shape=(512,512, len(temp_dat)), dtype=np.int16)
        new_fn = []
        for cnt, idx in enumerate(np.argsort(temp_idx)):
            new_dat[..., cnt] = temp_dat[idx]
            new_fn += [fn[idx]]
        return new_dat, new_fn

    dcm_fn = sorted(glob(os.path.join(in_dir, "*.dcm")), key=os.path.basename)
    dat, cur_fn = load_dcm_data(fn=dcm_fn)

    return dat, cur_fn

def normalizePlanes(npzarray, transpose=True, clip=False, is_old=False):
    maxHU = 1200.
    minHU = 15
    if is_old:
        maxHU = 400.
        minHU = -1000.
    # maxHU = np.percentile(npzarray, 90)
    # minHU = np.percentile(npzarray, 10)
    if transpose:
        npzarray = np.transpose(npzarray, (0, 2, 3, 1))
    npzarray = npzarray.astype(cur_dtype)
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    if clip:
        npzarray[npzarray > 1] = 1.
        npzarray[npzarray < 0] = 0.
    return npzarray


dat, fn = load_data()

deeplab_model = Deeplabv3(input_shape=(512, 512, 3), classes=3, activation="softmax")
deeplab_model.load_weights(ckpt)


base_idx = np.arange(dat.shape[-1])
rand_idx = np.tile(base_idx[..., None], (1, 3))
for d in range(3): rand_idx[..., d] += (d - 3 // 2)
rand_idx[rand_idx < 0] = 0
rand_idx[rand_idx >= dat.shape[-1]] = dat.shape[-1] - 1

all_res = np.zeros(dat.shape, dtype=np.uint8)
from tqdm import tqdm
for cnt, (bi, ri) in enumerate(tqdm(zip(base_idx, rand_idx), disable=False, total=len(base_idx))):
    res = deeplab_model(normalizePlanes(dat[None, ..., ri], transpose=False), training=False)
    res = np.argmax(res.numpy().squeeze(), axis=-1)
    Image.fromarray((res*127.5).astype(np.uint8)).save(fp=fn[cnt][:-4]+"_mask.png")
    Image.fromarray(((res==1)*127.5).astype(np.uint8)).save(fp=fn[cnt][:-4]+"_liv.png")
    Image.fromarray(((res==2)*255).astype(np.uint8)).save(fp=fn[cnt][:-4]+"_spl.png")

    all_res[...,cnt] = res
print("finished")