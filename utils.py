import config
import numpy as np

def load_segment_data():
    mm_mode = "r"
    trn_dat = np.load(config.raw_data_path + "segment/trn_dat.npy", mmap_mode=mm_mode)
    trn_lbl = np.load(config.raw_data_path + "segment/trn_lbl.npy", mmap_mode=mm_mode)
    trn_bound_mask = np.load(config.raw_data_path + "segment/trn_bound_mask_2d.npy", mmap_mode=mm_mode)
    trn_shp = np.load(config.raw_data_path + "segment/trn_shp.npy", mmap_mode=mm_mode)

    return trn_dat, trn_lbl, trn_shp, trn_bound_mask

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
    npzarray = npzarray.astype(np.float32)
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    if clip:
        npzarray[npzarray > 1] = 1.
        npzarray[npzarray < 0] = 0.
    return npzarray

def dice_coef(y_true, y_pred, smooth=1):
    # intersection = tf.keras.backend.sum(y_true * y_pred, axis=(1,2))
    # union = tf.keras.backend.sum(y_true + y_pred, axis=(1,2))
    # return tf.keras.backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    intersection = y_true * y_pred
    union = y_true + y_pred
    return (2. * intersection + smooth) / (union + smooth)

def get_seg_trn_idx(shp, msk):
    all_idx = []

    for cur_pat, cur_idx in enumerate(shp[:, 0]):
        temp_idx = np.arange(cur_idx[0] - 1, msk.shape[2] - cur_idx[1] + 1)
        temp_idx = np.tile(temp_idx[..., None], (1, config.seg_D))
        temp_idx += [-1, 0, 1]

        temp_idx[temp_idx < 0] = 0
        temp_idx[temp_idx >= msk.shape[2]] = msk.shape[2] - 1

        if cur_idx[0] == 0:
            temp_idx = temp_idx[1:-1]
        temp_idx = np.concatenate((np.full(shape=len(temp_idx), fill_value=cur_pat)[..., None], temp_idx), axis=-1)
        all_idx += [temp_idx]
    all_idx = np.concatenate(all_idx, axis=0)
    msk = np.transpose(msk, (1, 2, 0, 3)).squeeze()[..., 1:]
    cse = []
    for idx in all_idx:
        cur_msk = msk[idx[0], idx[1]]
        if cur_msk[0] == 0 and cur_msk[1] == 0:
            cur_case = 0
        elif cur_msk[0] == 1 and cur_msk[1] == 1:
            cur_case = 1
        elif cur_msk[0] == 1 and cur_msk[1] == 0:
            cur_case = 2
        elif cur_msk[0] == 0 and cur_msk[1] == 1:
            cur_case = 3
        cse += [cur_case]

    all_idx = np.concatenate((all_idx, np.array(cse)[..., None]), axis=-1)

    return all_idx
