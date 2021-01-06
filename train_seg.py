from model import Deeplabv3
import numpy as np
import os
import tensorflow as tf
import tqdm
import utils
import config

keras = tf.keras
K = keras.backend

class Trainer:
    def __init__(self, summ_path):
        self.summ_path = summ_path


    def train_one_batch(self, model, dat, lbl, b_mask, optim, vars):
        dat = utils.normalizePlanes(dat, clip=False)
        with tf.GradientTape() as grad_tape:
            predictions = model(dat, training=True)
            loss = tf.keras.losses.SparseCategoricalCrossentropy()(lbl, predictions)
            dice_loss_all = -utils.dice_coef(tf.keras.backend.one_hot(lbl, num_classes=3), predictions)
            dice_loss = tf.boolean_mask(dice_loss_all, b_mask)
            loss += K.mean(dice_loss)*3.
            # loss += tf.keras.losses.MeanSquaredError()(tf.keras.backend.one_hot(lbl, num_classes=3), predictions)
        g_grads = grad_tape.gradient(loss, vars)
        optim.apply_gradients(zip(g_grads, vars))
        return loss

    def logger(self, model, dat, lbl, b_mask, step, img=True):
        dat = utils.normalizePlanes(dat, clip=False)
        res = model(dat, training=False)
        if img:
            summ_img = dat[...,1]
            summ_img = ((summ_img-tf.reduce_min(summ_img))/tf.reduce_max(summ_img))
            summ_img = tf.concat((summ_img,tf.cast(tf.keras.backend.argmax(res, axis=-1), tf.float32)/2., lbl/2.), axis=-1)[...,None]
            summ_img = tf.reshape(summ_img, [1, -1, summ_img.shape[-2], summ_img.shape[-1]])
            tf.summary.image(name="result", data=summ_img, step=step)
            tf.summary.flush()
        one_hot_lbl = tf.keras.backend.one_hot(lbl, num_classes=3)
        CE_loss = tf.keras.metrics.SparseCategoricalCrossentropy()(lbl, res)
        MSE_loss = tf.keras.metrics.MeanSquaredError()(one_hot_lbl, res)
        dice_loss = K.mean(utils.dice_coef(one_hot_lbl, res), axis=(0,1,2))
        tf.summary.scalar('CE', CE_loss, step=step)
        tf.summary.scalar('MSE', MSE_loss, step=step)
        tf.summary.scalar('Dice_0', dice_loss[0], step=step)
        tf.summary.scalar('Dice_1', dice_loss[1], step=step)
        tf.summary.scalar('Dice_2', dice_loss[2], step=step)

    def train(self):
        deeplab_model = Deeplabv3(input_shape=(512, 512, config.seg_D), classes=3, activation="softmax")
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.00001,
            decay_steps=config.iter_cnt,
            decay_rate=0.999,
            staircase=True)
        optim = tf.keras.optimizers.Adam(lr_schedule)
        train_vars = deeplab_model.trainable_variables

        (trn_dat_2d, trn_mask_2d, trn_b_mask, trn_idx_2d, trn_code_2d, trn_stg_2d), (tst_dat_2d, tst_mask_2d, tst_b_mask, tst_idx_2d, tst_code_2d, tst_stg_2d), idx_to_str = utils.load_segment_data()

        trn_mask_idx = np.any(trn_mask_2d, axis=(-1,-2))

        if not os.path.exists(self.summ_path+"model/"):
            os.makedirs(self.summ_path+"model/")

        global_step = 0
        print("Started Training at ", self.summ_path)
        for cur_epoch in tqdm.trange(config.epoch_cnt, desc=self.summ_path.split("/")[-2]):
            (base_pat_id, base_idx), (rand_pat_id, rand_idx) = utils.get_seg_trn_idx(idx=trn_idx_2d, code=trn_code_2d, mask_idx=trn_mask_idx)
            for cur_step in tqdm.trange(0, len(rand_idx), config.batch_size):
                cur_dat = trn_dat_2d[rand_idx[cur_step:cur_step+config.batch_size]]
                cur_lbl = trn_mask_2d[base_idx[cur_step:cur_step+config.batch_size]]
                cur_b_mask = trn_b_mask[base_idx[cur_step:cur_step+config.batch_size]]
                self.train_one_batch(dat=cur_dat, lbl=cur_lbl, b_mask=cur_b_mask, model=deeplab_model, optim=optim, vars=train_vars)

                if global_step%10==0:
                    self.logger(model=deeplab_model, dat=cur_dat, lbl=cur_lbl, b_mask=cur_b_mask, step=global_step)
                global_step+=1
            tf.keras.models.save_model(deeplab_model, self.summ_path+"model/%03d"%cur_epoch)