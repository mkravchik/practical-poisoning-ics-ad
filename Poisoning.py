import os

import sys
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import time, datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from scipy import signal
import collections
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error
from math import sqrt
from munch import *
from tqdm import tqdm
import os
import glob
import re, copy, random
import seaborn as sns
import matplotlib as mpl
import pickle
from copy import deepcopy
from scipy import interpolate
import scipy.io

from keras.layers import Input, Dense, Dropout, Flatten, Reshape, Conv1D, MaxPooling1D
from keras.models import Model, load_model
from keras import optimizers
from keras.callbacks import *
import subprocess

data_path = "./"


def dumpGraph(g=None, title=''):
    print(title)
    if g is None:
        g = tf.get_default_graph()
    print(g)
    keys = g.get_all_collection_keys()
    if keys is None or 0 == len(keys):
        print("Empty graph")
    for k in keys:
        print(k)
        print(g.get_collection(k))


def show_weights(e_weight, vmin=None, vmax=None, title=""):
    if vmin is None:
        vmin = np.min([np.min(e_weight)])
    if vmax is None:
        vmax = np.max([np.max(e_weight)])
    plt.figure()
    im = plt.imshow(e_weight, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title(title + " weights mean %f std %f" % (np.mean(e_weight), np.std(e_weight)))
    plt.colorbar(im)


def sin_generator(c):
    periods = c["periods"]
    period1 = 1  # each second we come back to the original point
    t = np.linspace(0, periods, c["total_len"])
    signal_func = c["signal_func"]
    s = signal_func(2 * np.pi / period1 * t)
    return s.reshape((c["total_len"], 1)), t


def double_sin_generator(c):
    periods = c["periods"]
    period1 = 1  # each second we come back to the original point
    t = np.linspace(0, periods, c["total_len"])
    s1 = np.sin(2 * np.pi / period1 * t)
    s2 = s1 * 0.5  # np.power(s1, 2)#np.cos(2 * np.pi/period1*t)
    #     s3 = np.power(s1, 2)#np.cos(2 * np.pi/period1*t)#s1*0.5#
    src = np.empty((c["total_len"], 2))
    src[:, 0] = s1
    src[:, 1] = s2
    #     src[:,2] = s3
    return src, t


# In[4]:


class sin_generator_c(object):
    def __call__(self, c):
        periods = c["periods"]
        period1 = 1  # each second we come back to the original point
        t = np.linspace(0, periods, c["total_len"])
        signal_func = c["signal_func"]
        s = 0.5 * signal_func(2 * np.pi / period1 * t)
        return s.reshape((c["total_len"], 1)), t

    def apply(self, src, op, _):
        src = op(src)
        return src


class double_sin_generator_c(object):
    def __init__(self):
        self.ops = [lambda x: x * 0.5]

    def __call__(self, c):
        periods = c["periods"]
        period1 = 1  # each second we come back to the original point
        t = np.linspace(0, periods, c["total_len"])
        src = np.empty((c["total_len"], 2))
        src[:, 0] = 0.5 * c["signal_func"](2 * np.pi / period1 * t)
        src[:, 1] = self.ops[0](src[:, 0])
        return src, t

    def apply(self, src, op, partial=None):
        # in this simple case we always change the first sine
        res = src.copy()
        res[:, 0] = op(res[:, 0])
        if partial is None:  # else leave it as is
            res[:, 1] = self.ops[0](res[:, 0])
        return res


class four_sig_generator_c(object):
    def __init__(self):
        self.ops = [lambda x: x * 0.25, lambda x: x ** 2]

    def __call__(self, c):
        periods = c["periods"]
        period1 = 1  # each second we come back to the original point
        t = np.linspace(0, periods, c["total_len"])
        src = np.empty((c["total_len"], 4))
        src[:, 0] = 0.5 * c["signal_func"](2 * np.pi / period1 * t)
        src[:, 1] = self.ops[0](src[:, 0])
        src[:, 2] = self.ops[1](src[:, 0])
        src[:, 3] = src[:, 1] + src[:, 2] - 0.1

        return src, t

    def apply(self, src, op, partial=None):
        # in this simple case we always change the first sine
        res = src.copy()
        res[:, 0] = op(res[:, 0])
        if partial is None:  # else leave it as is
            res[:, 1] = self.ops[0](res[:, 0])
            res[:, 2] = self.ops[1](res[:, 0])
            res[:, 3] = res[:, 1] + res[:, 2] - 0.1
        return res


class echo_generator_c(object):
    def __init__(self, src_signal):
        self.sig = src_signal.copy()

    def __call__(self, c):
        periods = c["periods"]
        period1 = 1  # each second we come back to the original point
        t = np.linspace(0, periods, len(self.sig))
        return self.sig, t

    def apply(self, src, op, partial=None):
        if partial is not None:
            out_cols = np.delete(np.arange(src.shape[-1]), partial)
            res = np.empty(src.shape)
            res[:, partial] = op(src[:, partial])
            res[:, out_cols] = src[:, out_cols]
            return res
        else:
            src = op(src)
            return src.copy()


def test_generators():
    conf = dict(
        signal_func=np.sin,
        att_magn=0.5,
        att_point="BOTTOM",  # "TOP", "SIDE" #TODO - add an attack-generating function
        total_len=100,
        periods=5,
        generator=double_sin_generator_c(),
    )

    src, _ = conf["generator"](munchify(conf))
    plt.figure(figsize=(20, 10))
    for col in range(src.shape[-1]):
        plt.plot(src[:, col], label="Col %d" % col)
        plt.legend()

    # test echo generator
    echo = echo_generator_c(src)
    echo_sig, t = echo(conf)

    plt.figure(figsize=(20, 10))
    for col in range(echo_sig.shape[-1]):
        plt.plot(echo_sig[:, col], label="Col %d" % col)
        plt.legend()

    def update_sig(sig):
        sig[3] += 0.5
        return sig

    att_src = conf["generator"].apply(src, update_sig)
    plt.figure(figsize=(20, 10))
    for col in range(att_src.shape[-1]):
        plt.plot(att_src[:, col], label="Col %d" % col)
        plt.legend()

    # try partial echo generator
    att_src = echo.apply(src, update_sig, partial=[1])
    plt.figure(figsize=(20, 10))
    for col in range(att_src.shape[-1]):
        plt.plot(att_src[:, col], label="Col %d" % col)
        plt.legend()


# In[5]:


def model_signal(conf, model_name, src, restore=False):
    print("model_signal", conf, model_name, src.shape)
    random.seed(1)
    c = munchify(conf)
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
        if restore:
            saver = tf.train.import_meta_graph(model_name + ".meta")
        else:
            # once the new graph is set to be default, we can fix its seed
            tf.set_random_seed(1)
            # Using float64 instead of the default float32 gives much faster run and slightly better results
            x = tf.placeholder("float64", [None, c.seq_len, src.shape[-1]], name="x")
            if not c.silent: print("Adding x (input placeholder), shape", x.shape)

            training = tf.placeholder(tf.bool, shape=(), name="training")
            print("x:", x.shape)

            flat = tf.reshape(x, (-1, x.shape[-2] * x.shape[-1]), name="flat_x")
            if not c.silent: print("Adding flat_x (reshape), shape", flat.shape)

            if c.inflate_factor:
                inflate_size = x.shape[-2] * x.shape[-1] * c.inflate_factor
            else:
                inflate_size = x.shape[-2] * x.shape[-1]
            code_size = max(((x.shape[-2] * x.shape[-1]).value) // c.code_ratio, 1)
            sizes = np.linspace(inflate_size.value, code_size, c.layers + 1).astype(int)
            print(sizes)

            if c.inflate_factor:
                enc_l = tf.layers.dense(flat, inflate_size, tf.nn.tanh, name="encoder_inflator")
                if not c.silent: print("Adding encoder_inflator (dense), shape", enc_l.shape, "activation: tanh")
            else:
                enc_l = flat

            for l_i, layer_size in enumerate(sizes[1:]):
                enc_l = tf.layers.dense(enc_l, layer_size, tf.nn.tanh, name="encoder_%d" % l_i)
                if not c.silent: print("Encoder: Adding dense ", "encoder_%d shape" % l_i, enc_l.shape, "activation: tanh", layer_size)

            if c.get("batch_norm", False):
                enc_l = tf.layers.BatchNormalization()(enc_l)
                if not c.silent: print("Adding BatchNormalization")

            dec_l = enc_l
            if c.inflate_factor:
                dec_l = tf.layers.dense(dec_l, inflate_size, tf.nn.tanh, name="decoder_inflator")
                if not c.silent: print("Adding decoder_inflator (dense), shape", dec_l.shape, "activation: tanh")

            for l_i, layer_size in enumerate(sizes[:-1][::-1]):
                dec_l = tf.layers.dense(dec_l, layer_size, tf.nn.tanh, name="decoder_%d" % l_i)
                if not c.silent: print("Decoder: Adding dense ", "decoder_%d shape" % l_i, dec_l.shape, "activation: tanh", layer_size)

            # To add it correctly, I must add a training tensor and pass it to the feed dict
            # The problem is - I'm not sure what to do with the training tensor value
            # when calculating the back gradient....
            if c.get("dropout_rate", False):
                dec_l = tf.layers.dropout(dec_l, c.dropout_rate, training=training, name="dropout")
                if not c.silent: print("Adding dropout", c.dropout_rate)

            # it is important to NOT have any activation in the last layer
            # as the activations limit the output to their range
            dec_l = tf.layers.dense(dec_l, (x.shape[-2] * x.shape[-1]), name="decoder_last")
            if not c.silent: print("Decoder: Adding decoder_last (dense) shape",  dec_l.shape)

            if c.activate_last:
                dec_l = tf.nn.tanh(dec_l, name="decoder_last_activation")
                if not c.silent: print("Decoder: Adding decoder_last_activation (tanh) shape",  dec_l.shape)


            out = tf.reshape(dec_l, [-1, c.seq_len, src.shape[-1]], name="out")
            if not c.silent: print("Adding out (reshape) shape",  out.shape)

            diff = tf.subtract(x, out, name="diff")
            if c.get("predict_one", False):
                error = tf.reduce_mean(tf.square(diff[:, -1, :]), name="adv_error")
            else:
                error = tf.reduce_mean(tf.square(diff), name="adv_error")
            optimizer = c.optimizer
            train = optimizer.minimize(error, name='train')

            init = tf.global_variables_initializer()

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config, graph=g) as sess:
        if len(src.shape) == 2:
            batches = c.batches
            #             #round the length so that it is divisible by batch_size
            rounded_len = len(src) // batches * batches
            src = src[:rounded_len, :]
            org_shape = src.shape
            src = src.reshape((batches, rounded_len // batches, -1))
        else:
            # src will arrive as an array [train_points, steps, features]
            # and we will train on all of the samples simultaneously
            org_shape = src.shape
            pass

        if restore:
            saver.restore(sess, model_name)
        else:
            # Run the initializer
            sess.run(init)

        # Show what we have in the graph
        trainable_vars = tf.trainable_variables()
        weights = []
        for v in trainable_vars:
            if v.name.find('kernel') != -1:
                if not c.silent: print("Layer", v.name, v)
                weights.append(v)

        loop = tqdm(range(c.it_count))
        prev_error = 0
        min_delta = c.get("min_delta", 0.0001)
        patience = c.get("patience", 6)
        patience_cnt = 0
        for it in loop:
            err_acc = 0
            # run over all samples
            if "single_sequence" in c and c.single_sequence:
                err_acc, _ = sess.run(["adv_error:0", "train"], feed_dict={"x:0": src, "training:0": True})
            else:
                # should I add noise here as well?
                # split into batches for speedup
                num_seqs = src.shape[-2] - c.seq_len + 1
                #                 print("num_seqs",num_seqs, src.shape)
                # randomize
                # IMPORTANT - without randomization the last training
                # batch is pulling the weights in its
                # direction thus producing inferior prediction
                # as all my signals are aligned in form
                batch_indices = np.arange(num_seqs)
                if c.randomize:
                    random.shuffle(batch_indices)
                for b in batch_indices:
                    res = sess.run(["adv_error:0", "out:0", "train"],
                                   feed_dict={"x:0": src[:, b:b + c.seq_len, :], "training:0": True})
                    #                     print(res[0])
                    err_acc += res[0] / num_seqs

            # I want error to decrease
            if prev_error:
                delta = prev_error - err_acc
                if delta < min_delta:  # not decreasing enough or even increasing
                    patience_cnt += 1
                else:
                    patience_cnt = 0
            prev_error = err_acc

            if patience_cnt >= patience:
                print("Epoch %d, avg. error %f, early stopping" % (it, err_acc))
                break
            if not c.silent:
                if it % 10 == 0 or it == c.it_count - 1: print("Epoch %d, avg. error %f" % (it, err_acc))
            # TEMP

        # now run the test
        err_acc = 0
        # run over all samples
        if "single_sequence" in c and c.single_sequence:
            err_acc, n_out = sess.run(["adv_error:0", "out:0"], feed_dict={"x:0": src, "training:0": False})
        else:
            net_out = []
            num_seqs = src.shape[-2] - c.seq_len + 1
            #             print("num_seqs", num_seqs, src.shape)

            for b in range(num_seqs):
                res = sess.run(["adv_error:0", "out:0"],
                               feed_dict={"x:0": src[:, b:b + c.seq_len, :], "training:0": False})
                #                 print(res[0])
                err_acc += res[0] / num_seqs
                if b == 0:
                    if c.get("predict_one", False):
                        # take the beginning from the input and the last one from the prediction
                        net_out.append(src[:, :c.seq_len - 1, :])
                        net_out.append(res[1][:, -1:, :])
                    else:
                        net_out.append(res[1])
                else:
                    # take the last element only
                    net_out.append(res[1][:, -1:, :])

            n_out = np.concatenate(net_out, axis=1)
        if False:  # not c.silent:
            print("n_out.shape", n_out.shape, "org_shape", org_shape)
            n_out = n_out.reshape((org_shape))
            src = src.reshape((org_shape))
            # plotting only the first signal's prediction
            #             plt.figure(figsize=(20,10))
            if len(org_shape) == 2:
                src = src.reshape((1, -1, org_shape[-1]))
                n_out = n_out.reshape((1, -1, org_shape[-1]))

            roll_errors = []
            for sig_i in [0]:  # range(len(src)):
                diffs = np.abs(src[sig_i, :, :] - n_out[sig_i, :, :])
                for col in range(src.shape[-1]):
                    plt.figure(figsize=(20, 10))
                    plt.plot(src[sig_i, :, col], label="Original col %i" % col)
                    plt.plot(n_out[sig_i, :, col], label="Prediction col %i" % col)
                    plt.legend()
                    roll_diffs = pd.Series(diffs[:, col]).rolling(c.window, min_periods=1).min()
                    col_alerts = (roll_diffs > c.threshold).sum()
                    roll_errors.append(roll_diffs.max())
                    if col_alerts:
                        print("Signal %d still raises %d alerts" % (sig_i, col_alerts))
                plt.title("Trained Model Test Sig %d, error %s" % (sig_i, err_acc))

            # show the weights
            #             for w in weights:
            #                 wv = sess.run(w)
            #                 show_weights(wv, title = repr(w))
            #                 print(w.name, np.mean(wv), np.std(wv))

            print("Max model error", np.max(np.abs(src - n_out)), "max rolling error", np.max(roll_errors))
        saver = tf.train.Saver()
        saver.save(sess, model_name)
        # tf.reset_default_graph()

        return err_acc, it


def test_model_signal():
    conf = dict(
        seq_len=2,
        total_len=100,
        code_ratio=2,
        signal_func=np.sin,  # signal.square,#signal.sawtooth#np.sin
        it_count=300,
        lr=0.6,
        layers=1,  # increasing it did not show any significant changes,
        inflate_factor=2,
        silent=False,
        batches=1,
        train_points=10,
        periods=5,
        optimizer=tf.train.GradientDescentOptimizer(0.6),
        # tf.train.MomentumOptimizer(0.6, 0.5)#tf.train.AdamOptimizer(0.05) #c.lr
        generator=double_sin_generator_c(),
        single_sequence=False,
        randomize=False,
        activate_last=False,
        window=1,
        threshold=0.2
    )
    model_name = "./simple_signal_seq_%d_batches_%d" % (conf["seq_len"], conf["batches"])
    np.random.seed(1)
    noise = np.random.normal(0, 0.05, (conf["train_points"], conf["total_len"], 1))
    src, _ = conf["generator"](munchify(conf))
    noised_src = noise + src
    print(noised_src.shape)
    plt.figure(figsize=(20, 10))
    for sig_i in range(len(noised_src)):
        for col in range(noised_src.shape[-1]):
            plt.plot(noised_src[sig_i, :, col], label="Sig %d Col %d" % (sig_i, col))
            plt.legend()

    # Interesting: with last layer activation lr 0.6 provides best of the 3 modes, without - 0.2  (for period 1)
    # for period 5 I can't get to max error less than 0.2 when randomize is on with the last layer activation!
    # for conf["activate_last"] in [False]:
    #     for conf["lr"] in [0.006, 0.2, 0.6]:
    #         for conf["randomize"] in [False]:
    #             conf["optimizer"] = tf.train.GradientDescentOptimizer(conf["lr"])
    #             model_signal(conf, model_name, noised_src)

    # model_signal(conf, model_name, src, noise, restore = True)


# In[6]:


def show_model_weights(model_name):
    weights = dict()
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            tf.train.import_meta_graph(model_name + ".meta").restore(sess, model_name)
            trainable_vars = tf.trainable_variables()
            for v in trainable_vars:
                if v.name.find('kernel') != -1:
                    print("Layer", v.name, v)
                    vv = sess.run(v)
                    #                     print(vv)
                    show_weights(vv, title=v.name)
                weights[v.name] = vv
    return weights


def reshape_and_plot_multi_2(data_title_list, ylim=(-1, 1)):
    fig, axes = plt.subplots(nrows=data_title_list[0][0].shape[-1], ncols=1,
                             figsize=(20, 5 * data_title_list[0][0].shape[-1]))
    for col in range(data_title_list[0][0].shape[-1]):
        ax = axes[col] if type(axes) == np.ndarray else axes
        for data, title in data_title_list:
            ax.plot(data.reshape((-1, data.shape[-1]))[:, col], label=title + " column %d" % col)
        ax.set_ylim(ylim[0], ylim[1])
        ax.legend()
    plt.show()
    plt.close()


def reshape_and_plot_multi(data, title="", ylim=(-1, 1)):
    fig, axes = plt.subplots(nrows=data.shape[-1], ncols=1, figsize=(20, 5 * data.shape[-1]))
    for col in range(data.shape[-1]):
        ax = axes[col] if type(axes) == np.ndarray else axes
        ax.plot(data.reshape((-1, data.shape[-1]))[:, col], label="Column %d" % col,
                #                      marker=all_markers[ps_idx%len(all_markers)]
                )
        ax.set_ylim(ylim[0], ylim[1])
        ax.legend()
    plt.title(title)
    plt.show()
    plt.close()


def poison_desc(poison, reference):
    return mean_squared_error(poison.reshape((-1, poison.shape[-1])),
                              reference.reshape((-1, poison.shape[-1])), squared=False)


def test_input(sess, x_i, c, display=False, title=""):
    mpl.rcParams.update({'font.size': 12})
    if "single_sequence" in c and c.single_sequence:
        n_out, test_err = sess.run(["out:0", "adv_error:0"], feed_dict={"x:0": x_i, "training:0": False})
    else:
        x_i = x_i.copy().reshape((1, -1, x_i.shape[-1]))
        net_out = []
        err_acc = 0
        for b in range(x_i.shape[-2] - c.seq_len + 1):
            res = sess.run(["adv_error:0", "out:0"], feed_dict={"x:0": x_i[:, b:b + c.seq_len, :], "training:0": False})
            err_acc += res[0]
            if b == 0:
                if c.get("predict_one", False):
                    # take the beginning from the input and the last one from the prediction
                    net_out.append(x_i[:, :c.seq_len - 1, :])
                    net_out.append(res[1][:, -1:, :])
                else:
                    net_out.append(res[1])
            else:
                # take the last element only
                net_out.append(res[1][:, -1:, :])

        test_err = err_acc / (x_i.shape[-2] - c.seq_len + 1)
        n_out = np.concatenate(net_out, axis=1)

    n_out_flat = n_out.reshape((-1, n_out.shape[-1]))
    x_i_flat = x_i.reshape((-1, x_i.shape[-1]))
    diffs = np.abs(x_i_flat - n_out_flat)
    alerts = 0
    roll_errors = []
    for col in range(n_out.shape[-1]):
        roll_diffs = pd.Series(diffs[:, col]).rolling(c.window, min_periods=1).min()
        col_alerts = (roll_diffs > c.threshold).sum()
        roll_errors.append(roll_diffs.max())
        #         if col_alerts:
        #             print(col_alerts, "in column ", col)

        alerts += col_alerts

    #     alerts = np.sum(np.abs(x_i - n_out) > c.threshold)
    # save the data so we can visualize it later as we wish
    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    # pd.DataFrame(x_i_flat, columns=c.field_names).to_csv("Poisoning_real_%d_"%alerts + timestamp + ".csv")
    # pd.DataFrame(n_out_flat, columns=c.field_names).to_csv("Poisoning_predicted_%d_"%alerts + timestamp + ".csv")
    if display:
        size = 16
        sns.set(rc={"font.size": size, "axes.titlesize": size, "axes.labelsize": size, "legend.fontsize": 12,
                'xtick.labelsize': size, 'ytick.labelsize': size}, style="darkgrid", palette='bright', context='talk')

        for col in range(n_out.shape[-1]):
            fig, _ = plt.subplots(1, 1)
            residue = np.abs(x_i_flat[:, col] - n_out_flat[:, col])
            alert = pd.Series(residue).rolling(c.window, min_periods=1).min() > c.threshold
            plt.plot(x_i_flat[:, col], label="Real")
            plt.plot(n_out_flat[:, col], label="Predicted", lw=0.75)
            #fig.axes[0].fill_between(alert.index, alert, facecolor='red')
            plt.plot(residue, label="Residue", lw=0.75)
            plt.plot(np.ones(len(x_i_flat)) * c.threshold, label="Threshold", linestyle='dashed', lw=0.5)
            plt.plot(alert, label="Alert", color='r', lw=0.75)
            plt.legend()
            plt.ylabel("Signal")
            plt.xlabel("Timepoint")
            plt.yticks(np.arange(-1.6, 1.6, step=0.2))
            plt.title(title)
            plt.savefig("Poisoning_" + timestamp + "_sig_%d.pdf" % col, dpi=600, bbox_inches='tight')
            plt.close()

    return test_err, n_out, alerts, np.max(roll_errors)


def build_attack_signal(input_sig, c):
    poisoned_point = 0
    x_i_att = copy.copy(input_sig[0]) + c.att_magn
    x_poison = copy.copy(input_sig[0])
    for i in range(2, len(input_sig)):
        # Changing the first -1 is an easy attack
        # changing a point that is on the top is hard - no solution
        # changing one at the bottom - works as well
        if c.att_point == "BOTTOM":
            cond = (input_sig[i][0] == -1 and input_sig[i - 1][0] == -1)
        elif c.att_point == "TOP":
            cond = (input_sig[i][0] == 1 and input_sig[i - 1][0] == 1)
        elif c.att_point == "SIDE":
            cond = (input_sig[i][0] == -1)
        elif c.att_point == "SIN_BOTTOM":
            cond = (np.abs(input_sig[i][0] + 0.5) <= 0.01)
        elif c.att_point == "SIN_TOP":
            cond = (np.abs(input_sig[i][0] - 0.5) <= 0.01)
        elif c.att_point == "SIN_SIDE":
            cond = (np.abs(input_sig[i][0]) <= 0.1)
        elif c.att_point == "CUSTOM" or c.att_point == "CUSTOM_FIXED" or c.att_point == "CUSTOM_LINEAR":
            cond = (i == c.att_start)
        else:
            cond = True
        if cond:
            partial = None if not "partial_attacked" in c or 0 == len(c["partial_attacked"]) else c["partial_attacked"]
            x_poison = copy.copy(input_sig[i:i + c.att_len, :])
            x_i_att = copy.copy(input_sig[i:i + c.att_len, :])

            def update_sig(sig):
                if c.att_point == "CUSTOM_FIXED":
                    sig = c.att_magn
                elif c.att_point == "CUSTOM_LINEAR":
                    addition = np.linspace(0, c.att_magn, c.att_len).reshape((-1, 1))
                    sig += addition
                else:
                    sig += c.att_magn
                return sig

            x_i_att = c.generator.apply(x_i_att, update_sig, partial)
            poisoned_point = i
            break

    return x_i_att, x_poison, poisoned_point


def generate_poison_and_attack(conf):
    print("generate_poison_and_attack")
    c = munchify(conf)
    input_sig, t = c.generator(c)

    input_len = len(input_sig)
    x_i_att, x_poison, poisoned_point = build_attack_signal(input_sig, c)

    x_att_full = copy.copy(input_sig)
    x_att_full[poisoned_point:poisoned_point + c.att_len, :] = x_i_att

    return input_sig.copy(), x_att_full


def test_generate_poison_and_attack():
    conf = dict(
        signal_func=np.sin,
        att_magn=0.5,
        att_len=2,
        att_point="SIN_BOTTOM",  # "TOP", "SIDE" #TODO - add an attack-generating function
        total_len=100,
        periods=2,
        generator=sin_generator_c(),
    )

    start_poison, x_att = generate_poison_and_attack(conf)

    plt.figure(figsize=(20, 10))
    for col in range(start_poison.shape[-1]):
        plt.plot(start_poison[:, col], label="start_poison Col %d" % col)
        plt.plot(x_att[:, col], label="x_att Col %d" % col)
        plt.legend()


# I want to check training and testing in the original setting, without wrapping
# I arrived to the conclusion that I need to retrain (?!) despite the waste of time
# otherwise, especially in the single_sequence setting, the training only with poison skewes the
# weights towards the attack, and screwes the clean data unrepearably

def train_and_test_adv(model_name, model_idx, it_count, x_train, complete_poison,
                       x_poison, x_test, c, train_curr_poison=True,
                       display=False, title=""):
    # print("train_and_test_adv:", model_name, model_idx, it_count, x_train.shape, len(complete_poison),
    #                   x_poison.shape, x_test.shape, c, train_curr_poison, display, title)
    if c.randomize:
        random.seed()
    else:
        random.seed(1)

    #     if not c.silent:
    #         print("train_and_test_adv x_train", x_train.shape)
    #         if len(complete_poison):
    #             print("complete_poison[0]", complete_poison[0].shape)
    def train_and_test_adv_internal_keras(model_name, model_idx, it_count, x_train, complete_poison,
                                          x_poison, x_test, c, train_curr_poison=True,
                                          display=False, title=""):
        print("train_and_test_adv_internal_keras !!!!!!!!!!")

        train_err = test_err = alerts = max_err = val_alerts = 0
        model_file_name = model_name
        n_out = []

        # we receive the inputs with 3 dimensions: batches, timesteps, features
        # flatten them 
        x_train = x_train.reshape((-1, x_train.shape[-1]))
        x_poison = x_poison.reshape((-1, x_poison.shape[-1]))
        x_test = x_test.reshape((-1, x_test.shape[-1]))

        # build combined input
        inp_srcs = [x_train]
        if len(complete_poison):
            for p in complete_poison:
                inp_srcs.append(p.reshape((-1, p.shape[-1])))
        if train_curr_poison:
            inp_srcs.append(x_poison)

        inp = np.concatenate(inp_srcs)
        print("x_train.shape", x_train.shape, len(complete_poison), "inp.shape", inp.shape, "x_test", x_test.shape)

        sliced_X_inp, sliced_Y_inp = get_ae_sequences(inp, inp, c.seq_len, c.out_seq_len, lag=0)
        x_val, _ = c.generator(c)
        sliced_X_val, sliced_Y_val = get_ae_sequences(x_val, x_val, c.seq_len, c.out_seq_len, lag=0)
        sliced_X_att, sliced_Y_att = get_ae_sequences(x_test, x_test, c.seq_len, c.out_seq_len, lag=0)

        print(type(np.array(sliced_X_inp)), np.array(sliced_X_inp).shape, np.array(sliced_Y_inp).shape)
        print(type(np.array(sliced_X_val)), np.array(sliced_X_val).shape, np.array(sliced_Y_val).shape)
        print(type(np.array(sliced_X_att)), np.array(sliced_X_att).shape, np.array(sliced_Y_att).shape)

        # restore and train the predictor
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        tf.reset_default_graph()
        with tf.Graph().as_default() as g:
            with tf.Session(graph=g, config=config).as_default():
                model = load_model(model_file_name)
                if it_count:
                    # need to train the model again....
                    earlyStopping = EarlyStopping(monitor='val_loss', patience=3,
                                                  verbose=1 if not c.silent else 0,
                                                  min_delta=1e-4, mode='auto')
                    lr_reduced = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1,
                                                   verbose=1 if not c.silent else 0,
                                                   min_delta=1e-4, mode='min')

                    # initialize time
                    start_time = time.time()

                    # train the model with the poison data
                    history = model.fit(np.array(sliced_X_inp), np.array(sliced_Y_inp),
                                        epochs=500,
                                        batch_size=32,
                                        shuffle=True,
                                        callbacks=[earlyStopping, lr_reduced],
                                        verbose=2 if not c.silent else 0,
                                        validation_data=(np.array(sliced_X_val), np.array(sliced_Y_val)))

                # make the prediction on the attack data
                Y_att_hat = model.predict(np.array(sliced_X_att))
                del model

        Y_att = np.concatenate(sliced_Y_att)
        Y_att_hat = Y_att_hat.reshape((-1, Y_att_hat.shape[-1]))

        diffs = np.abs(Y_att - Y_att_hat)
        alerts = 0
        roll_errors = []
        for col in range(Y_att_hat.shape[-1]):
            roll_diffs = pd.Series(diffs[:, col]).rolling(c.window, min_periods=1).min()
            col_alerts = (roll_diffs > c.threshold).sum()
            roll_errors.append(roll_diffs.max())
            alerts += col_alerts

        # for col in range(Y_att_hat.shape[-1]):
        #     plt.figure()
        #     plt.plot(Y_att[:, col], label="Attack Input")
        #     plt.plot(Y_att_hat[:, col], label="Poisoned Attack Output")
        #     residue = np.abs(Y_att[:, col] - Y_att_hat[:, col])
        #     plt.plot(residue, label="Residue")
        #     plt.plot(np.ones(len(Y_att_hat)) * c.threshold, label="Threshold")
        #     alert = pd.Series(residue).rolling(c.window, min_periods=1).min() > c.threshold
        #     plt.plot(alert, label="Alert")
        #     plt.legend()
        #     plt.ylabel("Signal")
        #     plt.xlabel("Timepoint")
        #     plt.yticks(np.arange(-1.6, 1.6, step=0.2))
        #     plt.title(title)
        #     timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        #     plt.savefig("ConvDetector_" + timestamp + "_sig_%d.pdf" % col, dpi=600,  bbox_inches='tight')
        #     plt.close()

        return train_err, test_err, Y_att_hat, alerts, model_file_name, np.max(roll_errors), val_alerts

    def train_and_test_adv_internal(model_name, model_idx, it_count, x_train, complete_poison,
                                    x_poison, x_test, c, train_curr_poison=True,
                                    display=False, title=""):
        if c.get("keras", False):
            # TEST
            train_err, test_err, n_out, alerts, model_file_name, max_err, val_alerts = train_and_test_adv_internal_keras(
                model_name, model_idx, 0, x_train, [],
                x_poison, x_test, c, False, display, "Test without poison")
            print("Keras Test without poison", train_err, test_err, alerts, max_err, val_alerts)

            return train_and_test_adv_internal_keras(model_name, model_idx, it_count, x_train, complete_poison,
                                                     x_poison, x_test, c, train_curr_poison, display, title)
        if x_train is not None:
            if len(x_train.shape) == 3 and x_train.shape[0] != c.batches:
                #       if not c.silent: print("Reshaping from ", x_train.shape,
                #                               "into %d batches"% c.batches)
                # will be reshaped by the next line
                x_train = x_train.reshape((-1, x_train.shape[-1]))

            if len(x_train.shape) == 2:
                # With real signals I have training_points * att_len * features, so if I reshape it to
                # batches * att_len * features it can't be concatenated with the poison, which is 1 * att_len * features
                #             batches = c.batches
                #     #             #round the length so that it is divisible by batch_size
                #             rounded_len = len(x_train)//batches * batches
                #             x_train = x_train[:rounded_len,:]
                #             x_train = x_train.reshape((batches, rounded_len//batches, -1))
                # TODO - test with all other configurations
                att_len = x_poison.shape[-2]
                rounded_len = len(x_train) // att_len * att_len
                x_train = x_train[:rounded_len, :]
                x_train = x_train.reshape((-1, att_len, x_train.shape[-1]))

        #     if not c.silent:
        #         print("train_and_test_adv x_train reshaped", x_train.shape)

        model_file_name = model_name + "_poison_%d" % (len(complete_poison))
        err_acc = 0
        with tf.Graph().as_default():
            tf.set_random_seed(1)
            config = tf.ConfigProto(log_device_placement=False)
            config.gpu_options.allow_growth = True
            train_err = 0
            with tf.Session(config=config) as sess:
                if it_count:  # we have some training to do
                    tf.train.import_meta_graph(model_name + ".meta").restore(sess, model_name)

                    # build combined input
                    inp_srcs = [x_train]
                    if len(complete_poison):
                        inp_srcs.append(np.array([p[0, :, :] for p in complete_poison]))
                    if train_curr_poison:
                        inp_srcs.append(x_poison)

                    # there is problem with the form of batches
                    inp = np.concatenate(inp_srcs)
                    #                 if not c.silent: print("inp", inp.shape, "it_count", it_count)

                    prev_error = 0
                    min_delta = c.get("min_delta", 0.0001)
                    patience = c.get("patience", 6)
                    patience_cnt = 0

                    for it in range(it_count):
                        if "single_sequence" in c and c.single_sequence:
                            train_err, _ = sess.run(["adv_error:0", "train"],
                                                    feed_dict={"x:0": inp, "training:0": True})
                        else:
                            train_err = 0
                            err_acc = 0
                            num_seqs = x_train.shape[-2] - c.seq_len + 1
                            batch_indices = np.arange(num_seqs)
                            if c.randomize:
                                random.shuffle(batch_indices)
                            for b in batch_indices:
                                err, _ = sess.run(["adv_error:0", "train"],
                                                  feed_dict={"x:0": inp[:, b:b + c.seq_len, :], "training:0": True})
                                err_acc += err
                            train_err = err_acc / (x_train.shape[-2] - c.seq_len + 1)

                            # I want error to decrease
                            if prev_error:
                                delta = prev_error - train_err
                                if delta < min_delta:  # not decreasing enough or even increasing
                                    patience_cnt += 1
                                else:
                                    patience_cnt = 0
                            if patience_cnt >= patience:
                                print("train_and_test_adv epoch %d, avg. error %f, prev_error %f, early stopping" % (
                                it, train_err, prev_error))
                                break
                            prev_error = train_err
                #                     if not c.silent:
                #                       print("train_and_test_adv it %d error %f" % (it, train_err))
                else:  # no training, restore the weights
                    tf.train.import_meta_graph(model_name + ".meta").restore(sess, model_name)

                # Run on the entire input
                test_err, n_out, alerts, max_err = test_input(sess, x_test, c, display, title)

                val_alerts = 0
                val_err = 0
                x_val, _ = c.generator(c)
                x_val = x_val.reshape((1, x_val.shape[-2], x_val.shape[-1]))
                val_err, _, val_alerts, _ = test_input(sess, x_val, c, False, title + " for validation ")

                # We are going to save and restore later to be wrapped
                model_file_name = model_name + "_%d" % model_idx
                tf.train.Saver().save(sess, model_file_name)

        return train_err, test_err, n_out, alerts, model_file_name, max_err, val_alerts

    if c.get("sec_seq_len") is None:
        return train_and_test_adv_internal(model_name, model_idx, it_count, x_train, complete_poison,
                                           x_poison, x_test, c, train_curr_poison, display, title)
    else:
        #         if not c.silent: print("Running with seq_len", c.seq_len)
        res1 = train_and_test_adv_internal(model_name, model_idx, it_count, x_train, complete_poison,
                                           x_poison, x_test, c, train_curr_poison, display, title)

        #         if not c.silent: print("Running with sec_seq_len", c.sec_seq_len)
        tmp = c.seq_len
        c.seq_len = c.sec_seq_len
        res2 = train_and_test_adv_internal(c.sec_model_name, model_idx, it_count, x_train, complete_poison,
                                           x_poison, x_test, c, train_curr_poison, display, title)
        c.seq_len = tmp
        # if not c.silent:
        #     print("res1", res1)
        #     print("res2", res2)
        # return the biggest error
        if res1[5] > res2[5]:
            #             if not c.silent: print ("returning", res1)
            return res1
        else:
            #             if not c.silent: print ("returning", res2)
            return res2


def back_gradient_optimization(sess, x_poison, it_count, epsilon, lr, adv_lr,
                               t_grad_x, weights, t_weights_grads,
                               weights_vals, weights_grads_vals, mask=None, bo_max_norm=True):
    if mask is None:
        mask = np.ones(x_poison.shape)
    # Rollback the iterations
    dxp = np.zeros(x_poison.shape)

    for it in range(it_count):
        # %% Line 4 g_(t-1) <= grad(L(x_c, w_t))
        # Get weights' gradients on the poison input
        # [c,dww] = getDerivativesMLP(x2,y2,net);
        # TODO - if I'm going to use dropout I need to figure out the value for training
        t_weights_grads_ = sess.run(t_weights_grads, feed_dict={"x:0": x_poison})
        t_weights_poison_grads = np.array([a[0] for a in t_weights_grads_])

        # Use a trick to calculate the hessian-vector product without calculating it
        ###############   The trick ################################################
        # wwm = wwm + 0.5.*epsilon.*dw;
        for w_i, w in enumerate(weights):
            sess.run(tf.assign(w, weights_vals[w_i] + 0.5 * epsilon * weights_grads_vals[w_i]))

            # %% Derivative of error relative to x
        # [c2x,dw2x] = getDerivativesMLP2(x2,y2,net2);
        # %% Derivative of error relative to w
        # [c2,dw2] = getDerivativesMLP(x2,y2,net2);
        [g_x_p] = sess.run([t_grad_x], feed_dict={"x:0": x_poison})
        g_x_p = g_x_p[0]
        t_weights_grads_ = sess.run(t_weights_grads, feed_dict={"x:0": x_poison})
        t_weights_poison_grads_plus = np.array([a[0] for a in t_weights_grads_])

        # wwm = wwm - epsilon.*dw;
        for w_i, w in enumerate(weights):
            sess.run(tf.assign(w, weights_vals[w_i] - 0.5 * epsilon * weights_grads_vals[w_i]))

            # %% Derivative of error relative to x
        # [c2x,dw2x] = getDerivativesMLP2(x2,y2,net2);
        # %% Derivative of error relative to w
        # [c2,dw2] = getDerivativesMLP(x2,y2,net2);
        [g_x_m] = sess.run([t_grad_x], feed_dict={"x:0": x_poison})
        g_x_m = g_x_m[0]
        t_weights_grads_ = sess.run(t_weights_grads, feed_dict={"x:0": x_poison})
        t_weights_poison_grads_minus = np.array([a[0] for a in t_weights_grads_])

        # ddxp = (dw2x - dw1x)./epsilon;
        ddxp = (g_x_p - g_x_m) / epsilon

        # ddw = (dw2 - dw1)./epsilon;
        ddw = (t_weights_poison_grads_plus - t_weights_poison_grads_minus) / epsilon
        ############## End of Trick ##############################################################

        # %% Line 2 dx_c = dx_c - alpha * ...
        # dxp = dxp - alpha.*ddxp'; %'
        # NOTE - we rely upon knowing the LR!!!
        dxp = dxp - lr * ddxp

        # %% Line 3 dw = dw - alpha * ...
        # dw = dw - alpha.*ddw;
        weights_grads_vals -= lr * ddw

        # %% Line 5 w_(t-1) = w_t + alpha*g_(t-1)
        # For the next iteration
        # ww = ww + alpha.*dww;
        weights_vals += lr * t_weights_poison_grads
        #         weights_vals = np.array(weights_vals) + lr*t_weights_poison_grads

        for w_i, w in enumerate(weights):
            sess.run(tf.assign(w, weights_vals[w_i]))

            # This is present in the algorithm and absent from the code?!
    # Add the gradient of the input
    # dxp += grad_x[0]

    #####################################################################
    # Calculate the max of the absolute gradient values.
    # This is used to calculate the step-size.
    if bo_max_norm:
        grad_absmax = np.abs(dxp).max()

        # If the gradient is very small then use a lower limit,
        # because we will use it as a divisor.
        if grad_absmax < 1e-10:
            grad_absmax = 1e-10

        step_size = adv_lr / grad_absmax if grad_absmax > adv_lr else 1

        # reshape_and_plot_multi(dxp, "dxp after optimization")
        # apply partial mask
        dxp *= mask
        # print("dxp", dxp, "grad_absmax", grad_absmax)
        # I want to minimize the error thus -
        x_poison -= step_size * dxp
    else:
        # print("calculated dxp", dxp, "grad_absmax", np.abs(dxp).max())
        dxp = dxp / np.linalg.norm(dxp)
        # print("normalized dxp", dxp)
        dxp *= mask
        x_poison -= adv_lr * dxp

    return x_poison, dxp


def prepare_graph_for_optimization(sess, model_file_name, c, silent=True, input_len=1, xdims=1):
    if "single_sequence" in c and c.single_sequence:
        tf.train.import_meta_graph(model_file_name + ".meta").restore(sess, model_file_name)

        x = tf.get_default_graph().get_operation_by_name('x').outputs[-1]

        error = tf.get_default_graph().get_operation_by_name('adv_error').outputs[-1]

    else:  ############### WRAPPING ############################################
        x = tf.placeholder("float64", [1, input_len, xdims], name="x")
        x_exp = tf.expand_dims(x, 0)
        x_patches = tf.extract_image_patches(images=x_exp, ksizes=[1, 1, c.seq_len, 1],
                                             strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        x_patches = tf.squeeze(x_patches)
        x_patches = tf.reshape(x_patches, (-1, c.seq_len, xdims))

        # restore and bind
        tf.train.import_meta_graph(model_file_name + ".meta", input_map={"x:0": x_patches}).restore(sess,
                                                                                                    model_file_name)
        out_patches = tf.get_default_graph().get_operation_by_name("out").outputs[-1]
        out_reordered = out_patches[:, -1, :]

        out_restored = tf.reshape(out_reordered, (1, -1, xdims))

        # diff = tf.subtract(
        #     x[:, -out_restored.shape[-2].value:, :],
        #     out_restored[:, :, :])

        ## TEST: Taking into account partial attack!
        ## Minimizing error in the controlled columns only,
        ## Assuming that we should focus on bringing these closer to our goal,
        ## and the uncontrolled actuators will react accordingly
        attacked_cols = tf.constant(c.get("partial_attacked", list(np.arange(xdims))))

        if c.get("predict_one", False):
            minimized_true = tf.gather(x[:, -out_restored.shape[-2].value:, :], attacked_cols, axis=-1)
            minimized_pred = tf.gather(out_restored[:, :, :], attacked_cols, axis=-1)
        else:
            # I calculate the loss over the entire sequence so I should do the same for the optimization
            # diff = tf.subtract(x_patches, out_patches)
            minimized_true = tf.gather(x_patches, attacked_cols, axis=-1)
            minimized_pred = tf.gather(out_patches, attacked_cols, axis=-1)

        diff = tf.subtract(minimized_true, minimized_pred)

        # Using reduce_sum instead of reduce_mean provides larger gradients
        error = tf.reduce_mean(tf.square(diff)) + tf.reduce_max(tf.abs(diff))

    # add gradients to the graph
    t_grad_x = tf.gradients(error, x)
    t_weights_grads = []
    trainable_vars = tf.trainable_variables()

    weights = []
    for v in trainable_vars:
        if v.name.find('kernel') != -1:
            if not silent: print("Adding", v.name, v)
            weights.append(v)

    for w in weights:
        t_weights_grads.append(tf.gradients(error, w))

    return t_grad_x, weights, t_weights_grads, error


# In[10]:


def poison_model(conf, model_name, start_poison, x_att, x_train):
    # set to error to see where it comes from
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    # remove old intermediate models
    for f in glob.glob("." + os.sep + "tmp" + os.sep + model_name + '_*.*'):
        os.remove(f)

    c = munchify(conf)
    #     if not c.silent:
    #         #show the unpoisoned weights
    #         org_weights = show_model_weights(model_name)

    xdims = start_poison.shape[-1]
    input_len = len(start_poison)

    x_poison = start_poison.copy().reshape((1, -1, xdims))
    x_i_att = x_att.copy().reshape((1, -1, xdims))
    if len(x_train.shape) == 2:
        x_train = x_train.reshape((1, -1, xdims))

    x_i = x_poison.copy()  # not in use

    all_markers = list(Line2D.markers.keys())

    # Testing incomplete training
    it_count = c.adv_it_count
    errors = []
    failed_updates = 0
    orig_poison = x_poison.copy()
    prev_x_poison = x_poison.copy()

    complete_poison = []

    adv_lr = c.adv_lr
    orig_adv_lr = adv_lr
    loop = tqdm(range(c.max_adv_iter))
    poisons = []
    model_file_name = model_name

    # partial mask
    mask = None
    if "partial_attacked" in c and len(c["partial_attacked"]):
        mask = np.zeros_like(x_i_att)
        for col in c["partial_attacked"]:
            mask[:, :, col] = 1
    attacked_cols = c.get("partial_attacked", list(np.arange(x_i.shape[-1])))

    # controller predictor
    if conf.get("pred_seq_len") is not None:
        pred_model_file_name = "./tmp/te_attack_%d_pred_seq_%d_bs_%d.h5" % (conf["att_num"], conf["pred_seq_len"], conf["pred_bs_len"])
    
    out_mask = list(set(np.arange(x_att.shape[-1])) - set(conf["partial_attacked"]))

    i = -1
    train_err, test_err, n_out, alerts, _, max_error, val_alerts = \
        train_and_test_adv(model_name, i, c.it_count, x_train,
                           complete_poison, x_poison, x_i_att, c, train_curr_poison=False, display=False,
                           title="Test with NO poison")

    if not c.silent: print("No-poison Test: tr. error %f, test error %f %d alerts max err %f %d val alerts" % \
                           (train_err, test_err, alerts, max_error, val_alerts))

    tolerance = 5
    prev_adv_errors = collections.deque(2 * [1000000000], tolerance)
    # Make a small step in the direction of the attack trying to break out.
    break_out_rate = 1
    use_bo = True
    for i in loop:
        poisons.append(x_poison)
        prev_x_poison = x_poison.copy()
        start_train = time.time()
        if use_bo:
            used_bo = True
            # if not c.silent: print("\t\t\t\t\t\t\t\tTraining took", time.time() - start_train)
            ############ WRAPPING ##############################################################
            # Re-Define a placeholder that will receive a single-batch long input
            grad_backprop_start = time.time()
            tf.reset_default_graph()
            tf.set_random_seed(1)
            config = tf.ConfigProto(log_device_placement=False)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                t_grad_x, weights, t_weights_grads, error = \
                    prepare_graph_for_optimization(sess, model_file_name, c, i != 0 or c.silent, input_len, xdims)

                # For 32 bits I can use only 0.0001, smaller values result in 0 second gradients
                # However, I haven't noticed any improvement in results with smaller epsilons
                epsilon = 1e-8
                fetch = [t_grad_x, error]
                weights_idx = len(fetch)
                fetch.extend(weights)
                weights_grad_idx = len(fetch)
                fetch.extend(t_weights_grads)

                # Get the final gradients, weights and error on the (full) attack input
                # should I assess it on the attack + validation set?!
                # TODO - if I'm going to use dropout - consider what should the value for training be
                res = sess.run(fetch, feed_dict={"x:0": x_i_att})

                grad_x = res[0]
                #             print("grad_x.shape", len(grad_x), grad_x[0].shape, grad_x[0])
                #             reshape_and_plot_multi(grad_x[0], "grad_x before optimization")

                #             for i in range(weights_idx, weights_grad_idx):
                #                 print(i, res[i].shape)

                full_error = res[1]
                weights_vals = np.array(res[weights_idx:weights_grad_idx])
                weights_grads_vals = np.array([a[0] for a in res[weights_grad_idx:]])

                # Check if the error with the current poison is decreasing
                eps = -0.000001
                prev_adv_lr = adv_lr
                adv_lr_adjusted = False
                bo_max_norm = c.get("bo_max_norm", True)
                x_poison, dxp = back_gradient_optimization(sess, x_poison, it_count, epsilon, c.lr, adv_lr,
                                                           t_grad_x, weights, t_weights_grads,
                                                           weights_vals, weights_grads_vals, mask, bo_max_norm)

                # print("np.abs(dxp).max()", np.abs(dxp).max())
        else:
            print("The dxp is low, using interpolation instead")
            step_size = break_out_rate * (x_i_att - prev_x_poison) / 2
            x_poison = prev_x_poison + step_size
            if not c.silent:
                print("Changed the current poison point towards the attack.", \
                      poison_desc(orig_poison, x_poison), "Break out rate:", break_out_rate)
            used_bo = False

        # I'm changing all poisons PASSED TO THE DETECTOR to their predictions
        # returns only the predicted field
        # controller predictor
        if conf.get("pred_seq_len") is not None:
            _, Y_att_hat = load_cross_predict_np(pred_model_file_name, x_poison[0, :, :],
                                                 conf["partial_attacked"], out_mask, conf["pred_seq_len"], 1)

            # reshape_and_plot_multi(x_poison[:, :, out_mask], "before prediction")
            # why +1?
            x_poison[0, conf["pred_seq_len"] - 1:, out_mask] = Y_att_hat.T
            # reshape_and_plot_multi(x_poison[:, :, out_mask], "after prediction")

        # reshape_and_plot_multi_2([(orig_poison, "Original poison"),
        #                           (prev_x_poison, "Previous poison"),
        #                           (x_poison, "New poison"),
        #                           (x_i_att, "Attack")])

        poison_dist = poison_desc(orig_poison, x_poison)

        #         #if not c.silent: print("\t\t\t\t\t\t\t\tGradient backprop took", time.time() - grad_backprop_start)
        # let's check if this poison will cause an alert!
        poison_test_start = time.time()
        _, p_err, _, alerts, _, _, val_alerts = train_and_test_adv(model_name, i, c.it_count, x_train,
                                                                   complete_poison, x_poison, x_poison, c,
                                                                   train_curr_poison=False,
                                                                   display=False,
                                                                   title="Iteration %d. New poison: " % i)
        if alerts:
            if not c.silent: print("The new poison (%f) will cause " % \
                                   poison_desc(orig_poison, x_poison),
                                   alerts, "alerts and %d val_alerts" % val_alerts, "Poison error", p_err,
                                   "previous poison",
                                   poison_desc(orig_poison, prev_x_poison))
            if np.max(np.abs(prev_x_poison - orig_poison)):  # don't add zero poison
                if not c.silent:
                    print("Adding another poison point.", \
                          poison_desc(orig_poison, prev_x_poison))
                complete_poison.append(prev_x_poison)

            if used_bo:
                # if the poison creates alerts, its' too agressive - decrease the learning rate
                failed_updates += 1
                # speed up the step decreasing if we keep on failing
                adv_lr *= (0.90 ** failed_updates)
                if adv_lr <= 0.00001:
                    if not c.silent: print("The learning rate (%f) is already small, exiting!"
                                           % (adv_lr))
                    break
            else:  # used interpolation
                break_out_rate *= 0.9
                if break_out_rate <= 0.00001:
                    if not c.silent: print("The break_out_rate (%f) is already small, exiting!"
                                           % (break_out_rate))
                    break
            # let's try with the last successful poison
            x_poison = prev_x_poison.copy()  # copy.copy(input_sig)
            continue
        if val_alerts:
            if not c.silent: print("The new poison (%f) will cause " % \
                                   poison_desc(orig_poison, x_poison), "%d val_alerts" % val_alerts, \
                                   "Poison error", p_err, "previous poison", poison_desc(orig_poison, prev_x_poison))
            while val_alerts and len(complete_poison) < 200:
                # IMPORTANT - TO RUN REGRESSION TESTS.
                complete_poison.append(x_train[-1:, -start_poison.shape[-2]:, :])
                if not c.silent: print("Adding clean data.")
                _, p_err, _, alerts, _, _, val_alerts = train_and_test_adv(model_name, i, c.it_count, x_train,
                                                                           complete_poison, x_poison, x_poison, c,
                                                                           train_curr_poison=False,
                                                                           display=False)
            # keep this poison and move to the testing
            if len(complete_poison) >= 200:
                if not c.silent: print("Too many poison points %d, exiting!"
                                       % (complete_poison))
                break
        else:  # no alerts on poison? restore the learning rate, let's try to be aggressive
            failed_updates = 0
            break_out_rate = 1
            if used_bo:
                adv_lr /= 0.98  # orig_adv_lr
                adv_lr = np.clip(adv_lr, 0, 1)
            else:
                # new location - new luck!
                adv_lr = orig_adv_lr
            use_bo = True  # no alerts? back to BO!

        # if not c.silent: print("\t\t\t\t\t\t\t\tPoison testing took", time.time() - poison_test_start)
        # test
        # Using the full number of iterations allowed poisoning!
        train_cnt = 0
        while train_cnt < 3:
            train_err, test_err, n_out, alerts, model_file_name, max_err, val_alerts = \
                train_and_test_adv(model_name, i, c.it_count, x_train, complete_poison, x_poison, x_i_att, c,
                                   display=False)
            if train_err > 0.2:
                print("Got weird train error. Redo")
            else:
                break
        n_out_flat = n_out.reshape((-1, n_out.shape[-1]))
        x_test_flat = x_i_att.reshape((-1, x_i_att.shape[-1]))
        test_mse = mean_squared_error(x_test_flat[:, attacked_cols], n_out_flat[:, attacked_cols])

        if not c.silent:
            print("Iteration %d, poison %f, training error %f, \
            test error %f, %d alerts full err %f adv_lr %f, max err %f, val_alerts %d, test_mse on predicted cols %f" %
                  (i, poison_dist, train_err, test_err, alerts, full_error, adv_lr, max_err, val_alerts, test_mse))
            print("complete_poison", [poison_desc(orig_poison, p) for p in complete_poison])
        if not alerts + val_alerts:
            # double check again
            train_err, test_err, n_out, alerts, model_file_name, max_err, val_alerts = \
                train_and_test_adv(model_name, i, c.it_count, x_train, complete_poison, x_poison, x_i_att, c,
                                   display=False)
            print("Final poisoned training error %f, test error %f, %d alerts, %d val alerts, max_err %f" % \
                  (train_err, test_err, alerts, val_alerts, max_err))
            if not alerts + val_alerts:
                if not c.silent: print("We are done!!!")
                break
            else:
                if not c.silent: print("Hmm!!! Continue")
        elif val_alerts:
            while val_alerts and len(complete_poison) < 200:
                complete_poison.append(
                    x_train[-1:, -start_poison.shape[-2]:, :])  # The amount of added points can be adjustable
                if not c.silent: print("Adding clean data.")
                train_err, test_err, _, alerts, model_file_name, max_err, val_alerts = \
                    train_and_test_adv(model_name, i, c.it_count, x_train, complete_poison, x_poison, x_i_att, c)
                if not c.silent: print("%d alerts adv_lr %f max err %f val_alerts %d" %
                                       (alerts, adv_lr, max_err, val_alerts))
                if not alerts + val_alerts:
                    if not c.silent: print("We are done!!!")
                    break
        # check whether we are progressing. If not - add the poison
        if np.sum([test_mse < el for el in prev_adv_errors]) == 0:
            if not c.silent:
                print("No progress, seems we reached a plato. Adding the current poison point.", \
                      poison_desc(orig_poison, x_poison))
            complete_poison.append(x_poison.copy())
            if conf.get("use_breakout", False):
                use_bo = False  # switch to the interpolation trying to break out

        else:
            # push it to the queue
            prev_adv_errors.appendleft(test_mse)

    if c.get("alt_model_name", False):
        train_err, test_err, n_out, alerts, model_file_name, max_err, val_alerts = \
            train_and_test_adv(c.alt_model_name, i, c.it_count, x_train, complete_poison, x_poison, x_i_att,
                               c.alt_model_params,
                               display=False)

        print(
            "Final Test with Alt model: tr. error %f, test error %f, %d alerts max err %f %d val alerts" % (
                train_err, test_err, alerts, max_error, val_alerts))

    # Final result
    # if not c.silent: print("Got ", len(complete_poison) + 1, " poison points. Doing final check")
    #
    # train_err = 0
    # #         print("All poison", complete_poison, x_poison)
    # train_err, test_err, n_out, alerts, model_file_name, max_err, val_alerts = \
    #     train_and_test_adv(model_name, i, c.it_count, x_train,
    #                        complete_poison, x_poison, x_i_att, c, True, True, title="Final test")

    # print("Final poisoned training error %f, test error %f, %d alerts, %d val alerts, max_err %f" % \
    #      (train_err, test_err, alerts, max_err, val_alerts))

    # for col in range(n_out.shape[-1]):
    #     plt.figure()
    #     plt.plot(start_poison[:, col], label="Original nput", color='b')
    #     for ps_idx, ps in enumerate(complete_poison):
    #         plt.plot(ps.reshape((-1, ps.shape[-1]))[:, col], label="Poisoning input %d" % ps_idx,
    #                  #                      marker=all_markers[ps_idx%len(all_markers)]
    #                  )
    #     plt.plot(x_poison.reshape((-1, x_poison.shape[-1]))[:, col], label="Last poisoning input", color='r')
    #     plt.plot(x_i_att.reshape((-1, x_i_att.shape[-1]))[:, col], label="Attack input", color="g")
    #     plt.legend()
    #     plt.ylabel("Signal")
    #     plt.xlabel("Timepoint")
    #     plt.title("Poisoning Points")
    #
    # for col in range(n_out.shape[-1]):
    #     plt.figure()
    #     plt.plot(start_poison[:, col], label="Clean Input", color='b')
    #     for ps_idx, ps in enumerate(poisons):
    #         plt.plot(ps.reshape((-1, ps.shape[-1]))[:, col], label="Poisoning Input %d" % ps_idx,
    #                  #                      marker=all_markers[ps_idx%len(all_markers)]
    #                  )
    #     plt.plot(x_i_att.reshape((-1, x_i_att.shape[-1]))[:, col], label="Attack input", color="g")
    #     plt.legend()
    #     plt.title("Poisoning Points History")

    #     if not c.silent:
    #         #show the poisoned weights and the difference
    #         print("Poisoned model", model_file_name)
    #         new_weights = show_model_weights(model_file_name)
    #         for w_name in org_weights:
    #             #TODO - ranges
    #             show_weights(new_weights[w_name] - org_weights[w_name], title = "Difference for " + w_name)
    complete_poison.append(x_poison)
    return alerts + val_alerts, len(complete_poison), i, model_file_name, complete_poison


################################################################
def test_poison_model():
    conf = dict(
        adv_lr=0.05,
        seq_len=2,
        total_len=50,
        code_ratio=2,
        signal_func=np.sin,  # signal.square,#signal.sawtooth#np.sin
        att_magn=0.35,
        att_point="SIN_BOTTOM",  # "SIN_TOP", "SIN_SIDE"
        att_len=1,
        window=1,
        threshold=0.2,
        layers=1,  # increasing it did not show any significant changes
        inflate_factor=2,
        silent=True,
        batches=1,
        it_count=40,
        adv_it_count=40,
        lr=0.6,
        train_points=10,
        max_adv_iter=200,
        periods=5,
        randomize=False,
        optimizer=tf.train.GradientDescentOptimizer(0.6),
        # tf.train.MomentumOptimizer(0.6, 0.5)#tf.train.AdamOptimizer(0.05) #c.lr
        generator=sin_generator_c(),
        single_sequence=True,
        naive=False,
        find_poison=False,
        retrain=False,
        retrain_points=10,
        max_clean_poison=100,
        activate_last=False
    )
    if conf["single_sequence"]:
        conf["seq_len"] = conf["total_len"]
    else:
        conf["window"] = conf["att_len"] = conf["seq_len"]

    model_name = "./tmp/simple_signal_seq_%d_batches_%d" % (conf["seq_len"], conf["batches"])

    conf["silent"] = False
    conf["find_poison"] = False
    # conf["single_sequence"] = False
    # conf["seq_len"] = 2
    src, _ = conf["generator"](munchify(conf))
    np.random.seed(1)
    noise = np.random.normal(0, 0.05, (conf["train_points"], conf["total_len"], 1))
    # TODO - apply proportionally?
    noised_src = noise + src

    model_signal(conf, model_name, noised_src)
    start_poison, x_att = generate_poison_and_attack(conf)
    # for col in range(start_poison.shape[-1]):
    #     plt.figure(figsize=(20,10))
    #     plt.plot(start_poison[:,col], label="Start Poison", color = 'b')
    #     plt.plot(x_att[:, col], label="Attack input", color = "g")
    #     plt.legend()
    #     plt.title("Poison for Column %d"%col)
    alerts = 0
    # if conf["find_poison"]:
    #     start_poison, alerts = find_start_poison(conf, model_name, x_att, noised_src)
    # if alerts:
    #     print("Failed to find start poison. Test configuration", conf, "\nResults: ", alerts, " alerts")
    # else:
    #     poison_model(conf, model_name, start_poison, x_att, noised_src)


"""
  param: start_poison - (length, features)
  param: x_att - (length, features)
"""


def poison_model_naive(conf, model_name, start_poison, x_att, x_train):
    print("poison_model_naive: x_att", len(x_att))
    c = munchify(conf)
    xdims = start_poison.shape[-1]
    x_poison = start_poison.copy().reshape((1, -1, xdims))
    x_i_att = x_att.copy().reshape((1, -1, xdims))
    if len(x_train.shape) == 2:
        x_train = x_train.reshape((1, -1, xdims))

    # remove old intermediate models
    for f in glob.glob("." + os.sep + "tmp" + os.sep + model_name + '_*.*'):
        os.remove(f)

    #     if not c.silent:
    #         #show the unpoisoned weights
    #         org_weights = show_model_weights(model_name)

    curr_poison = x_poison.copy()
    poisons = []
    rate = 1
    step_size = 1
    i = 0
    prev_poison = curr_poison.copy()
    loop = tqdm(range(c.max_adv_iter))
    #     it_count = c.adv_it_count
    it_count = c.it_count
    smallest_step = 0.001

    # How does the model look like without any poisoning?
    # controller predictor
    if conf.get("pred_seq_len") is not None:
        if conf.get("pred_bs_len") is not None:
            # pred_model_file_name = "./tmp/te_pred_seq_%d_bs_%d.h5" % (conf.get("pred_seq_len"), conf["pred_bs_len"])
            pred_model_file_name = "./tmp/te_attack_%d_pred_seq_%d_bs_%d.h5" % (conf["att_num"], conf["pred_seq_len"], conf["pred_bs_len"])
        else:
            pred_model_file_name = "./tmp/te_seq_%d_out_seq%d.h5" % (conf["pred_seq_len"], 1)

    out_mask = list(set(np.arange(x_att.shape[-1])) - set(conf["partial_attacked"]))

    train_err, test_err, n_out, alerts, _, max_error, val_alerts = train_and_test_adv(model_name, i, it_count, x_train,
                                                                                      poisons,
                                                                                      curr_poison, x_i_att, c,
                                                                                      train_curr_poison=False,
                                                                                      display=True,
                                                                                      title="Test with NO poison")
    if not c.silent: print(
        "No-poison Test: tr. error %f, test error %f rate %f, %d alerts max err %f %d val alerts %f" % (
            train_err, test_err, rate, alerts, max_error, val_alerts, np.max(np.abs(step_size))))

    for i in loop:
        # what if I remove this and just add more of the last poison, where do I end up with?
        # Second, should I update the validation sometimes? How can I accomodate for the normal drift?
        # I guess, I need to take the oldest train records from the training set sometimes...
        #         if np.max(np.abs(step_size)) <= 0.0000001 or len(poisons) >= 300:
        #             break
        if np.max(np.abs(step_size)) >= smallest_step or 0 == len(
                poisons):  # if we have no poisons found, disregard the lower limit
            step_size = rate * (x_i_att - prev_poison) / 2
            curr_poison = prev_poison + step_size

            # I'm changing all poisons PASSED TO THE DETECTOR to their predictions
            if conf.get("pred_seq_len") is not None:
                if conf.get("pred_bs_len") is not None:
                    _, Y_att_hat = load_cross_predict_np(pred_model_file_name, curr_poison[0, :, :],
                                                         conf["partial_attacked"], out_mask, conf["pred_seq_len"], 1, 0, 
                                                         x_i_att[0, :conf.get("pred_bs_len"), :])
                else:
                    _, Y_att_hat = load_cross_predict_np(pred_model_file_name, curr_poison[0, :, :],
                                                         conf["partial_attacked"], out_mask, conf["pred_seq_len"], 1, 0)

                # NOTE: there is an error here: if we use lag=0, we must add the last values (we could not predict), not the first one
                # curr_poison[0, conf["pred_seq_len"] - 1:, out_mask] = Y_att_hat.T
                curr_poison[0, : len(Y_att_hat), out_mask] = Y_att_hat.T

            # test whether the new poison raises alert
            train_err, test_err, _, alerts, _, max_error, val_alerts = train_and_test_adv(model_name, i, it_count,
                                                                                          x_train, poisons,
                                                                                          curr_poison, curr_poison, c,
                                                                                          train_curr_poison=False,
                                                                                          display=False,
                                                                                          title="Current poison test ")
            if not c.silent: print(
                "Poison %f tr. error %f, test error %f rate %f, %d alerts max err %f %d val alerts %f" % (
                    poison_desc(x_poison, curr_poison),
                    train_err, test_err, rate, alerts, max_error, val_alerts, np.max(np.abs(step_size))))
            if alerts:  # + val_alerts: #Ignoring the validation alerts for a while
                # descrease the step
                rate *= 0.9
                continue
        else:
            # just add the last working poison
            curr_poison = prev_poison
            # increase the step - we are on the border of the smallest addition
            rate /= 0.9
            step_size = np.max([np.max(np.abs(step_size)),
                                smallest_step])  # so we try the next time, the sign does not matter, as the step is recalculated each time
            if not c.silent: print("Trying last working poison %f " % poison_desc(x_poison, curr_poison))
        # the poison does not cause alerts, add it
        step_size = np.max([np.max(np.abs(step_size)), smallest_step])  # make sure we will continue the next iteration
        poisons.append(curr_poison)
        # test
        train_err, test_err, n_out, alerts, model_file_name, max_error, val_alerts = train_and_test_adv(model_name, i,
                                                                                                        it_count,
                                                                                                        x_train,
                                                                                                        poisons,
                                                                                                        curr_poison,
                                                                                                        x_i_att, c,
                                                                                                        train_curr_poison=False,
                                                                                                        display=False,
                                                                                                        title="Test with poison %.3f " % poison_desc(
                                                                                                            x_poison,
                                                                                                            curr_poison))
        if not c.silent: print(
            "Test: Poison %f tr. error %f, test error %f rate %f, %d alerts max err %f %d val alerts %f" % (
                poison_desc(x_poison, curr_poison),
                train_err, test_err, rate, alerts, max_error, val_alerts, np.max(np.abs(step_size))))
        # Ignoring the validation alerts for a while
        # TODO - REMOVE THIS, TEST ONLY
        val_alerts = 0
        if (alerts + val_alerts) == 0:
            # double check
            train_err, test_err, n_out, alerts, model_file_name, max_error, val_alerts = train_and_test_adv(model_name, i,
                                                                                                            it_count,
                                                                                                            x_train,
                                                                                                            poisons,
                                                                                                            curr_poison,
                                                                                                            x_i_att, c,
                                                                                                            train_curr_poison=False,
                                                                                                            display=True,
                                                                                                            title="Test with poison %.3f " % poison_desc(
                                                                                                                x_poison,
                                                                                                                curr_poison))
            if not c.silent: print(
                "Final test: Poison %f tr. error %f, test error %f rate %f, %d alerts max err %f %d val alerts %f" % (
                    poison_desc(x_poison, curr_poison),
                    train_err, test_err, rate, alerts, max_error, val_alerts, np.max(np.abs(step_size))))
            if (alerts + val_alerts) == 0:
                if not c.silent: print("We are done! Got %d poison points and no alerts" % len(poisons))
                break
            else:
                if not c.silent: print("Hmm! continue...")
            
        elif val_alerts:
            if not c.silent: print("Got %d val_alerts with poison. Adding clean data." % val_alerts,
                                   poison_desc(x_poison, curr_poison))
            # decrease the step
            rate *= 0.9
            del poisons[-1]  # remove the tried poison
            poisons.append(x_train[-1:, -start_poison.shape[-2]:, :])  # The amount of added points can be adjustable
            continue  # don't change the prev_poison
        else:
            # increase the step
            rate /= 0.9

        # add the poison
        if not c.silent: print("Added poison ", poison_desc(x_poison, curr_poison))
        prev_poison = curr_poison.copy()

        i += 1

    # final test - already done earlier
    if c.get("alt_model_name", False):
        train_err, test_err, n_out, alerts, model_file_name, max_error, val_alerts = train_and_test_adv(
            c.alt_model_name, i,
            it_count,
            x_train,
            poisons,
            curr_poison,
            x_i_att, c.alt_model_params,
            train_curr_poison=False,
            display=False,
            title="Test with poison %.3f " % poison_desc(
                x_poison,
                curr_poison))

        print(
            "Final Test with Alt model: Poison %f tr. error %f, test error %f rate %f, %d alerts max err %f %d val alerts %f" % (
                poison_desc(x_poison, curr_poison),
                train_err, test_err, rate, alerts, max_error, val_alerts, np.max(np.abs(step_size))))

    return alerts + val_alerts, len(poisons), i + 1, model_file_name, poisons


def test_poison_model_naive():
    conf = dict(
        adv_lr=0.05,
        seq_len=2,
        total_len=50,
        code_ratio=2,
        signal_func=np.sin,  # signal.square,#signal.sawtooth#np.sin
        att_magn=0.35,
        att_point="SIN_BOTTOM",  # "SIN_TOP", "SIN_SIDE"
        att_len=1,
        window=1,
        threshold=0.2,
        layers=1,  # increasing it did not show any significant changes
        inflate_factor=2,
        silent=True,
        batches=1,
        it_count=40,
        adv_it_count=40,
        lr=0.6,
        train_points=10,
        max_adv_iter=200,
        periods=5,
        randomize=False,
        optimizer=tf.train.GradientDescentOptimizer(0.6),
        # tf.train.MomentumOptimizer(0.6, 0.5)#tf.train.AdamOptimizer(0.05) #c.lr
        generator=sin_generator_c(),
        single_sequence=True,
        naive=False,
        find_poison=False,
        retrain=False,
        retrain_points=10,
        max_clean_poison=100
    )
    if conf["single_sequence"]:
        conf["seq_len"] = conf["total_len"]
    else:
        conf["window"] = conf["att_len"] = conf["seq_len"]

    model_name = "./tmp/simple_signal_seq_%d_batches_%d" % (conf["seq_len"], conf["batches"])

    conf["silent"] = False
    # conf["single_sequence"] = False
    # conf["seq_len"] = 2
    src, _ = conf["generator"](munchify(conf))
    np.random.seed(1)
    noise = np.random.normal(0, 0.05, (conf["train_points"], conf["total_len"], 1))

    # TODO - apply proportionally?
    noised_src = noise + src

    # model_signal(conf, model_name, noised_src)
    # start_poison, x_att = generate_poison_and_attack(conf)
    # alerts = 0
    # if conf["find_poison"]:
    #     start_poison, alerts = find_start_poison(conf, model_name, x_att)
    # if alerts:
    #     print("Failed to find start poison. Test configuration", conf, "\nResults: ", alerts, " alerts")
    # else:
    #     poison_model_naive(conf, model_name, start_poison, x_att, noised_src)


# In[13]:


def find_start_poison(conf, model_name, x_att, x_train):
    c = munchify(conf)
    # remove old intermediate models
    for f in glob.glob("." + os.sep + "tmp" + os.sep + model_name + '_*.*'):
        os.remove(f)

    c = munchify(conf)
    xdims = x_att.shape[-1]
    input_len = len(x_att)
    x_poison = x_att.copy().reshape((1, -1, xdims))

    all_markers = list(Line2D.markers.keys())

    prev_adv_errors = collections.deque(2 * [1000000000], 2)
    failed_updates = 0
    orig_poison = x_poison.copy()
    prev_x_poison = x_poison.copy()

    complete_poison = []

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    # Initial test
    train_err, test_err, n_out, alerts, _, max_err, _ = train_and_test_adv(model_name, 0, 0, x_train,
                                                                           [], x_poison,
                                                                           x_poison, c, False, not c.silent,
                                                                           title="Initial test")

    adv_lr = c.adv_lr
    orig_adv_lr = adv_lr
    ############ WRAPPING ##############################################################
    # Re-Define a placeholder that will receive a single-batch long input
    grad_backprop_start = time.time()
    tf.reset_default_graph()
    tf.set_random_seed(1)
    with tf.Session(config=config) as sess:
        if "single_sequence" in c and c.single_sequence:
            tf.train.import_meta_graph(model_name + ".meta").restore(sess, model_name)

            x = tf.get_default_graph().get_operation_by_name('x').outputs[-1]

            error = tf.get_default_graph().get_operation_by_name('adv_error').outputs[-1]
        else:
            x = tf.placeholder("float64", [1, input_len, xdims], name="x")
            x_exp = tf.expand_dims(x, 0)
            x_patches = tf.extract_image_patches(images=x_exp, ksizes=[1, 1, c.seq_len, 1],
                                                 strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
            x_patches = tf.squeeze(x_patches)
            x_patches = tf.reshape(x_patches, (-1, c.seq_len, xdims))

            # restore and bind
            tf.train.import_meta_graph(model_name + ".meta", input_map={"x:0": x_patches}).restore(sess, model_name)
            out_patches = tf.get_default_graph().get_operation_by_name("out").outputs[-1]
            out_reordered = out_patches[:, -1, :]

            out_restored = tf.reshape(out_reordered, (1, -1, xdims))

            diff = tf.subtract(
                x[:, -out_restored.shape[-2].value:, :],
                out_restored[:, :, :])

            # Using reduce_sum instead of reduce_mean provides larger gradients
            error = tf.reduce_mean(tf.square(diff)) + tf.reduce_max(tf.abs(diff))  # TRY adding reduce_max

        # add gradients to the graph
        t_grad_x = tf.gradients(error, x)

        fetch = [t_grad_x, error]

        loop = tqdm(range(c.max_adv_iter))
        for i in loop:
            # Get the gradient, the full error on the current poison
            res = sess.run(fetch, feed_dict={"x:0": x_poison})

            grad_x = res[0][0]
            full_error = res[1]
            poison_dist = poison_desc(orig_poison, x_poison)

            #             if not c.silent:
            #                 print("Iteration %d, poison %f full err %f adv_lr %f" %
            #                   (i, poison_dist, full_error, adv_lr))

            # Check if the error with the current poison is decreasing
            eps = -0.000001
            prev_adv_lr = adv_lr
            adv_lr_adjusted = False
            # just roll back to the previous working poison and recalculate with it using smaller lr.
            # ignoring this allowed convergence for seq_len=2 and att_magn = 0.3
            if np.sum([full_error - el < eps for el in prev_adv_errors]) == 0 and adv_lr > 0.00001:
                failed_updates += 1
                # speed up the step decreasing if we keep on failing
                adv_lr = prev_adv_lr * (0.90 ** failed_updates)
                adv_lr_adjusted = True
                x_poison = prev_x_poison.copy()
            #                 if not c.silent:
            #                     print("The error (%f) is too big. Trying with lr %f and poison %f. prev_x_poison %f" %
            #                       (full_error, adv_lr, poison_desc(orig_poison, x_poison), poison_desc(orig_poison,
            #  prev_x_poison)))
            #                     print("prev_adv_errors", prev_adv_errors)
            elif adv_lr <= 0.00001:
                if not c.silent: print(
                    "The errors keep growing (%f) and the learning rate (%f) is already small, exiting!"
                    % (full_error, adv_lr))
                break
            else:  # we are good
                # push the error to the queue
                prev_adv_errors.appendleft(full_error)
                failed_updates = 0
                prev_x_poison = x_poison.copy()
                # restore adv_lr
                adv_lr = orig_adv_lr

            # I want to minimize the error thus -
            # Calculate the max of the absolute gradient values.
            # This is used to calculate the step-size.
            grad_absmax = np.abs(grad_x).max()

            # If the gradient is very small then use a lower limit,
            # because we will use it as a divisor.
            if grad_absmax < 1e-10:
                grad_absmax = 1e-10

            step_size = adv_lr / grad_absmax

            x_poison -= step_size * grad_x

    # Final result
    #         print("All poison", complete_poison, x_poison)
    train_err, test_err, n_out, alerts, _, max_err, _ = train_and_test_adv(model_name, 0, 0, x_train,
                                                                           [], x_poison,
                                                                           x_poison, c, False, not c.silent,
                                                                           title="Final test")

    if not c.silent:
        print("Final poisoned training error %f, test error %f, %d alerts, max_err %f" % (
            train_err, test_err, alerts, max_err))
        for col in range(n_out.shape[-1]):
            plt.figure(figsize=(20, 10))
            plt.plot(x_att[:, col], label="Original Attack Input", color='b')
            plt.plot(x_poison.reshape((-1, x_poison.shape[-1]))[:, col], label="Closest Starting Poison Found",
                     color='r')
            plt.legend()
            plt.title("Poison for Column %d" % col)

    return x_poison.reshape(x_att.shape), alerts


def test_find_start_poison():
    conf = dict(
        adv_lr=0.05,
        seq_len=2,
        total_len=50,
        code_ratio=2,
        signal_func=np.sin,  # signal.square,#signal.sawtooth#np.sin
        att_magn=0.35,
        att_point="SIN_BOTTOM",  # "SIN_TOP", "SIN_SIDE"
        att_len=1,
        window=1,
        threshold=0.2,
        layers=1,  # increasing it did not show any significant changes
        inflate_factor=2,
        silent=True,
        batches=1,
        it_count=40,
        adv_it_count=40,
        lr=0.6,
        train_points=10,
        max_adv_iter=200,
        periods=5,
        randomize=False,
        optimizer=tf.train.GradientDescentOptimizer(0.6),
        # tf.train.MomentumOptimizer(0.6, 0.5)#tf.train.AdamOptimizer(0.05) #c.lr
        generator=sin_generator_c(),
        single_sequence=True,
        naive=False,
        find_poison=False,
        retrain=False,
        retrain_points=10,
        max_clean_poison=100
    )
    if conf["single_sequence"]:
        conf["seq_len"] = conf["total_len"]
    else:
        conf["window"] = conf["att_len"] = conf["seq_len"]

    model_name = "./tmp/simple_signal_seq_%d_batches_%d" % (conf["seq_len"], conf["batches"])

    conf["silent"] = False
    # conf["single_sequence"] = False
    # conf["seq_len"] = 2
    src, _ = conf["generator"](munchify(conf))
    np.random.seed(1)
    noise = np.random.normal(0, 0.05, (conf["train_points"], conf["total_len"], 1))
    # TODO - apply proportionally?
    noised_src = noise + src

    # model_signal(conf, model_name, noised_src)
    # _, x_att = generate_poison_and_attack(conf)

    # start_poison, alerts = find_start_poison(conf, model_name, x_att, noised_src)

    # for col in range(start_poison.shape[-1]):
    #     plt.figure(figsize=(20,10))
    #     plt.plot(start_poison[:,col], label="Start Poison", color = 'b')
    #     plt.plot(x_att[:, col], label="Attack input", color = "g")
    #     plt.legend()
    #     plt.title("Poison for Column %d"%col)

    # print("Got" ,alerts, "alerts")


def train_more_and_test(conf, model_name, src, train_noise, x_att):
    conf_copy = conf.copy()

    c = munchify(conf_copy)
    conf_copy["train_points"] = conf_copy["retrain_points"]
    model_signal(conf_copy, model_name, src, train_noise, restore=True)
    # now resize the attack input and test
    x_att = x_att.copy().reshape((1, -1, x_att.shape[-1]))
    train_err, test_err, n_out, alerts, _, _, val_alerts = train_and_test_adv(model_name, 0, 0, None, [],
                                                                              x_att, x_att, c,
                                                                              train_curr_poison=False, display=True,
                                                                              title="Attack results after more "
                                                                                    "training: ")
    print("Retraining test error %f, %d alerts" % (test_err, alerts))


# # What should and can I test?
# I need to show what influences the ability to create the poisoning
# 1. The percentage of poisoning points in the normal points.
# How to test it? I can create more then one attack sample in the poison
# In any case, the validation should include the attack and the normal data together
#
# 2. Should I add the noise to double sin tests as well? - Yes. Done
#
# 3. Attack amplitude. Should I make it signal-dependent? In the case of double sin there is a little point of
# setting the second one as high as the first. Seems - yes. This means rerunning the tests.
#
# 4. Randomized training. As randomization is not directly applicable to a time series, as the order matters,
# I will test instead additional training after posioning!
#
# 5. Naive vs. back-optimization
#
# 6. Transferability of optimizer (Maybe)
#
# 7. Amount of training data (train_points)
#
# 8. Attack location
#
# Adding validation to the tests has shown that I ***MUST*** do the training, and not just load the model!
# Otherwise, the poison drags the weights in the desired direction and the validation fails even on attacks as low as
#  0.4
#
# I don't know exactly how it will work with batches, but let's start with passing the train data (with noise) in
#


# Synthetic tests
def run_synthetic_test(conf):
    start_time = time.time()
    ts = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    model_name = "./tmp/synthetic_signal_seq_%s_%d_batches_%d" % (ts, conf["seq_len"], conf["batches"])
    # remove old intermediate models
    for f in glob.glob("." + os.sep + "tmp" + os.sep + model_name + '_*.*'):
        os.remove(f)

    np.random.seed(1)
    noise = np.random.normal(0, 0.02, (conf["train_points"], conf["total_len"], 1))
    src, _ = conf["generator"](munchify(conf))

    # TODO - apply proportionally?
    noised_src = noise + src

    model_signal(conf, model_name, noised_src)
    if conf.get("sec_seq_len"):
        tmp = conf["seq_len"]
        sec_model_name = "./tmp/synthetic_signal_seq_%d_batches_%d" % (conf["sec_seq_len"], conf["batches"])
        conf["sec_model_name"] = sec_model_name
        conf["seq_len"] = conf["sec_seq_len"]
        model_signal(conf, sec_model_name, noised_src)
        conf["seq_len"] = tmp

    if conf.get("alt_model_params"):
        conf["alt_model_params"]["batches"] = conf["batches"]
        if conf["alt_model_params"].get("conv", False):
            cols = ["sig%d" % i for i in range(noised_src.shape[-1])]
            df_noised_src = pd.DataFrame(noised_src.reshape((-1, noised_src.shape[-1])), columns=cols)
            # otherwise, previous default graph resets confuse Keras completely
            tf.reset_default_graph()
            config = tf.ConfigProto(log_device_placement=False)
            config.gpu_options.allow_growth = True
            with tf.Graph().as_default() as g:
                with tf.Session(graph=g, config=config).as_default():
                    model = create_and_train_predictor(df_noised_src, cols, seq_len=conf["alt_model_params"]["seq_len"],
                                                       out_seq_len=conf["alt_model_params"]["out_seq_len"],
                                                       horizon=0,
                                                       data_generator=get_ae_sequences, silent=False,
                                                       out_field_names=cols, layers=conf["alt_model_params"]["layers"],
                                                       conv=True, conv_filters=conf["alt_model_params"]["conv_filters"])
                    conf["alt_model_name"] = "./tmp/synthetic_signal_alt_conv_%s_seq_%d_batches_%d" % (
                    ts, conf["alt_model_params"]["seq_len"], conf["alt_model_params"]["batches"])
                    model.save(conf["alt_model_name"])
                    del model
        else:
            conf["alt_model_name"] = "./tmp/synthetic_signal_alt_seq_%s_%d_batches_%d.h5" % (
            ts, conf["alt_model_params"]["seq_len"], conf["alt_model_params"]["batches"])
            model_signal(conf["alt_model_params"], conf["alt_model_name"], noised_src)

    print(conf)
    start_poison, x_att = generate_poison_and_attack(conf)

    if conf["find_poison"]:
        start_poison, alerts = find_start_poison(conf, model_name, x_att, noised_src)
        if alerts:
            print("Failed to find start poison. Test configuration", conf, "\nResults: ", alerts, " alerts")
            return

    poison_func = poison_model_naive if conf["naive"] else poison_model

    alerts, poison_points, iterations, poisoned_model_file_name, poisons = poison_func(conf, model_name, start_poison,
                                                                                       x_att,
                                                                                       noised_src)
    print("Test configuration", conf, "\nResults: ", alerts,
          " alerts", poison_points, "poison_points", iterations, "iterations")

    #     if conf["retrain"]:
    #         train_more_and_test(conf, poisoned_model_file_name, src, noise, x_att)

    time_taken = time.time() - start_time
    print("Took ", time_taken)
    return alerts, poison_points, iterations, time_taken


def synthetic_test():
    conf = dict(
        adv_lr=0.2,
        seq_len=2,
        total_len=100,
        code_ratio=2,
        signal_func=np.sin,  # signal.square,#signal.sawtooth#np.sin
        att_magn=0.5,
        att_point="SIN_TOP",  # "SIN_BOTTOM",# "SIN_SIDE"
        att_len=1,
        window=1,
        threshold=0.2,
        layers=1,  # increasing it did not show any significant changes
        inflate_factor=2,
        silent=True,
        batches=1,
        it_count=30,
        adv_it_count=30,
        lr=0.6,
        train_points=10,
        max_adv_iter=300,
        periods=5,
        randomize=True,
        optimizer=tf.train.GradientDescentOptimizer(0.6),
        # tf.train.MomentumOptimizer(0.6, 0.5)#tf.train.AdamOptimizer(0.05) #c.lr
        generator=sin_generator_c(),
        single_sequence=False,
        naive=False,
        find_poison=False,
        retrain=False,
        retrain_points=10,
        max_clean_poison=100,
        activation=tf.nn.tanh,
        activate_last=True,
        predict_one=False,
        bo_max_norm=True,
        patience=30
    )

    points_per_period = 500
    conf["silent"] = False
    conf["single_sequence"] = False
    conf["att_len"] = 1
    conf["layers"] = 1
    conf["inflate_factor"] = 2
    conf["randomize"] = True
    conf["partial_attacked"] = [0]
    conf["att_start"] = 374  # 605 #TOP
    conf["att_len"] = 40
    conf["att_point"] = "CUSTOM_FIXED"  # "SIN_BOTTOM",# "TOP", "SIDE" #TODO - add an attack-generating function
    # conf["sec_seq_len"] = 32

    # Actually, with naive poisoning this should be the case, I guess
    # more layers require smaller lr
    #    optimizer = tf.train.GradientDescentOptimizer(
    #        0.001),  # tf.train.MomentumOptimizer(0.6, 0.5)#tf.train.AdamOptimizer(0.05) #c.lr

    cols = list(conf.keys()) + ['alerts', 'poison_points', 'iterations', 'time']
    df_results = pd.DataFrame(columns=cols)
    sns.set(rc={"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "legend.fontsize": 12}, style="darkgrid")

    for points_per_period in [500]:  # range(20,100,20):
        for conf["naive"] in [True, False]:
            for conf["generator"] in [double_sin_generator_c()]:
                for conf["att_point"] in ["CUSTOM_FIXED"]:
                    for conf["train_points"] in [10]:  # range(10, 150, 20):
                        for conf["seq_len"] in np.arange(2, 52, 10):
                            for conf["periods"] in [2]:
                                start_magn = -0.2
                                # if conf["seq_len"] == 22:
                                #     start_magn = -0.1
                                for conf["att_magn"] in np.arange(start_magn, 0.5, 0.1):
                                    for alt_opt in [tf.train.GradientDescentOptimizer(0.6)]:
                                        # [tf.train.MomentumOptimizer(0.6, 0.5), tf.train.AdamOptimizer(0.001) - used in the x-opt tests
                                        # 1. it looks that lr 0.6 can be poisoned, but smaller can't (with BO) - to verify and understand why
                                        # 2. To make the test with 0.1, where we required poison points
                                        # 3. To make the test with patience 6 or 7
                                        # More tests:
                                        # 1. Different optimizer used by the model 
                                        # 2. Different model!
                                        # 2.1 AE with different length
                                        # 2.2 1D Conv.
                                        # All of these - with Naive = True
                                        failed = 0
                                        test_cnt = 1
                                        for _ in range(test_cnt):
                                            conf["total_len"] = conf["periods"] * points_per_period
                                            if conf["single_sequence"]:
                                                conf["seq_len"] = conf["total_len"]
                                            conf["batches"] = conf["train_points"]

                                            # Alternative model
                                            if "alt_model_params" in conf:
                                                del conf["alt_model_params"]
                                            conf["alt_model_params"] = conf.copy()
                                            conf["alt_model_params"]["keras"] = True
                                            conf["alt_model_params"]["conv"] = True
                                            conf["alt_model_params"]["conv_filters"] = 16
                                            conf["alt_model_params"]["seq_len"] = 18
                                            conf["alt_model_params"]["out_seq_len"] = 1
                                            conf["alt_model_params"]["layers"] = 3

                                            alerts, poison_points, iterations, time = run_synthetic_test(conf)
                                            res = conf.copy()
                                            res['alerts'] = alerts
                                            res['poison_points'] = poison_points
                                            res['iterations'] = iterations
                                            res['time'] = time

                                            df_results = df_results.append(pd.DataFrame([res]), ignore_index=True)
                                            df_results.to_csv(datetime.datetime.now().strftime(
                                                data_path + "PoisoningTests_%d_%m_%Y_%H_%M_%S.csv"))
                                            if res['alerts']:
                                                failed += 1
                                                if failed > test_cnt // 2:
                                                    break
                                        # if failed > test_cnt // 2:
                                        #    break

    df_results.to_csv(datetime.datetime.now().strftime(data_path + "PoisoningTests_%d_%m_%Y_%H_%M_%S.csv"))


data_path = "./"


def read_swat_data_file(file_name):
    df_train_orig = pd.read_csv(data_path + file_name, parse_dates=['Timestamp'], index_col=0, dayfirst=True)
    df_train_orig = df_train_orig.ffill()
    df_train_orig.columns = df_train_orig.columns.str.strip()
    #     df_train_orig.index = df_train_orig.index.str.strip()
    df_train_orig.index = pd.to_datetime(df_train_orig.index, format='%d/%m/%Y %I:%M:%S %p')
    return df_train_orig


def read_swat():
    df_train_orig = read_swat_data_file("SWaT_Dataset_Normal_sub5.csv")
    df_test_orig = read_swat_data_file("SWaT_Dataset_Attack_sub5.csv")

    # removing bad fields dramatically increases the recall rate
    sensor_cols = [col for col in df_train_orig.columns if
                   col not in ['Timestamp', 'Normal/Attack', "AIT201", 'AIT202', 'AIT203', 'P201', 'AIT501', 'AIT503',
                               'AIT504', 'AIT402']]

    # scale sensor data
    scaler = MinMaxScaler((-0.5, 0.5))
    # cut the first unstable part
    df_train_orig = df_train_orig.iloc[85000 // 5:, :]

    X = pd.DataFrame(index=df_train_orig.index, columns=sensor_cols,
                     data=scaler.fit_transform(df_train_orig[sensor_cols]))
    Y = pd.DataFrame(index=df_test_orig.index, columns=sensor_cols,
                     data=scaler.transform(df_test_orig[sensor_cols]))

    # In[16]:

    time_format = '%m/%d/%Y %I:%M:%S %p'

    attacks = dict()

    # 3
    attacks[3] = dict(
        attack_start='12/28/2015  11:22:00 AM',
        attack_end='12/28/2015  11:28:22 AM',
        field_names=['LIT101', "FIT101", "MV101", "P101"],
        #     delta_hours = 0.6
    )
    # 7
    attacks[7] = dict(
        attack_start='12/28/2015  12:08:25 PM',
        attack_end='12/28/2015  12:15:33 PM',
        field_names=['LIT301', "FIT201", 'P302'],
        #     delta_hours = 1.15
    )
    # Attack 8: 28/12/2015 13:10:10	13:26:13
    attacks[8] = dict(
        attack_start='12/28/2015  01:10:10 PM',
        attack_end='12/28/2015  01:26:13 PM',
        field_names=['DPIT301', "P602"])

    # Attack 10 28/12/2015 14:16:20	14:19:00
    attacks[10] = dict(
        attack_start='12/28/2015  02:16:20 PM',
        attack_end='12/28/2015  02:19:00 PM',
        field_names=['FIT401', "P402"])

    # Attack 16 29/12/2015 11:57:25	12:02:00
    attacks[16] = dict(
        attack_start='12/29/2015  11:57:21 AM',
        attack_end='12/29/2015  12:02:00 PM',
        field_names=['LIT301', "FIT201", 'P302'])  # adding more related fields allows better model

    # 31 31/12/2015 22:05:34	22:11:40
    attacks[31] = dict(
        attack_start='12/31/2015  10:05:34 PM',
        attack_end='12/31/2015  10:11:40 PM',
        field_names=["LIT401", 'P302', 'P402', 'LIT301'])  # adding more related fields allows better model

    # 32 1/01/2016 10:36:00	10:46:00 LIT-301
    attacks[32] = dict(
        attack_start='01/01/2016  10:36:00 AM',
        attack_end='01/01/2016  10:46:00 AM',
        field_names=['LIT301', "FIT201", 'P302'])

    # #33 1/01/2016 14:21:12	14:28:35	LIT-101
    attacks[33] = dict(
        attack_start='01/01/2016  02:21:12 PM',
        attack_end='01/01/2016  02:28:35 PM',
        field_names=['LIT101', "FIT101", "MV101", "P101"])
    # field_names = sensor_cols[:5]

    # 36 1/01/2016 22:16:01	22:25:00	LIT-101
    attacks[36] = dict(
        attack_start='01/01/2016  10:16:01 PM',
        attack_end='01/01/2016  10:25:00 PM',
        field_names=['LIT101', "FIT101", "MV101", "P101"])

    # None of the above was able to poison

    # #38 2/01/2015 11:31:38	11:36:18	AIT-402, AIT-502 ? involves AIT402
    # attack_start = '01/02/2016  11:31:38 AM'
    # attack_end = '01/02/2016  11:36:18 AM'
    # field_names = [] #TOFIND

    # #39 2/01/2015 2/01/2015 11:43:48	11:50:28	FIT-401, AIT-502
    # attack_start = '01/02/2016  11:43:48 AM'
    # attack_end = '01/02/2016  11:50:28 AM'
    # field_names = [] #TOFIND
    return X, Y, attacks


def display_swat():
    X, Y, attacks = read_swat()
    for at in attacks:
        att_start = datetime.datetime.strptime(attacks[at]["attack_start"], time_format)
        att_end = datetime.datetime.strptime(attacks[at]["attack_end"], time_format)
        delta_hours = 3
        if "delta_hours" in attacks[at]:
            delta_hours = attacks[at]["delta_hours"]
        delta = datetime.timedelta(hours=delta_hours)
        field_names = attacks[at]["field_names"]
        axes = Y[att_start - delta:att_end + delta][field_names].plot(subplots=True, legend=True,
                                                                      title="Normalized sensor values for attack #%d"
                                                                            % at)
        s_delta = datetime.timedelta(seconds=2.5)
        for col_idx in range(len(field_names)):
            Y[att_start - s_delta:att_start + s_delta][field_names].iloc[:, col_idx].plot(ax=axes[col_idx],
                                                                                          marker=mpl.markers.CARETRIGHT,
                                                                                          markersize=10, color='r')
            Y[att_end - s_delta:att_end + s_delta][field_names].iloc[:, col_idx].plot(ax=axes[col_idx],
                                                                                      marker=mpl.markers.CARETLEFT,
                                                                                      markersize=10, color='r')

        print(at, "\n", Y[att_start - delta:att_end + delta][field_names].min(),
              Y[att_start - delta:att_end + delta][field_names].max())
        print("Attack len", len(Y[att_start:att_end]), len(Y[att_start - delta:att_end + delta]))
    # Y[field_names].plot(subplots=True, legend=True, figsize=(20,10))


# # Synthetic SWaT attacks
# I want to test synthetic attacks on the SWaT data, namely generate the attack on the top of a real SWaT signal
# I arrived to the conclusion that it is not feasible to find the "correct" good starting poison from the given data,
#  as evident from the find_closest_signal test.
#
# If I use my optimization-based algorithm it sometimes fails to find the starting poison, while there should exist one!
#
# Therefore I should synthesize attacks similar to the SWaT ones and test them. This should be better then nothing,
# for sure.
#
# I should be able to imitate most of the attacks.
#
#
# | Attack | Imitation | Notes |
# | ------ | --------- | ------|
# | 3 | requires linear increase of LIT101 to 0.824721 | (add new feature) |
# | 7 | LIT301 fixed at 1.309346 | |
# | 8 | DPIT301 fixed at 1.689609 | There are changes in P602 that should disclose the attack |
# | 10 | FIT401 fixed at  -23.096798 | The signal is not periodic and AE does not work well with it |
# | 16 | LIT301 linearly decreases to -1.393334 |  add new feature |
# | 31 | LIT401 fixed at -1.269128 | It would be good to find a related feature |
# | 32 | LIT301 fixed at 1.313708 | |
# | 33 | LIT101 fixed at -1.081831 | |
# | 36 | LIT101 fixed at -1.232907| |
#

# In[ ]:


def find_closest_signal(find_in, to_find, eps=0):
    # assuming [points, features] shape
    # I'm sure I can make it more efficient
    min_dist = 100000000000
    min_idx = -1
    for start_idx in range(len(find_in) - len(to_find)):
        dist = np.linalg.norm(find_in[start_idx:start_idx + len(to_find), :] - to_find)
        if dist < min_dist:
            min_dist = dist
            min_idx = start_idx
            if min_dist < eps:
                break
    return min_idx


def run_synthetic_swat_test(conf, x_train_full, extended_att_len):
    start_time = time.time()
    model_name = "./tmp/synthetic_swat_seq_%d" % (conf["seq_len"])
    # remove old intermediate models
    for f in glob.glob("." + os.sep + "tmp" + os.sep + model_name + '_*.*'):
        os.remove(f)

    np.random.seed(1)
    conf["total_len"] = conf["train_points"] * extended_att_len
    x_train = x_train_full[-conf["total_len"]:]
    period_start = conf["period_start"] if "period_start" in conf else 0
    # it will be used for generating a clean validation input
    conf["generator"] = echo_generator_c(x_train_full[period_start:extended_att_len])

    model_signal(conf, model_name, x_train_full)

    start_poison, x_att = generate_poison_and_attack(conf)
    if conf["find_poison"]:
        start_poison, alerts = find_start_poison(conf, model_name, x_att, x_train)
        if alerts:
            print("Failed to find start poison. Test configuration", conf, "\nResults: ", alerts, " alerts")
            return alerts, 0, 0, 0

    poison_func = poison_model_naive if conf["naive"] else poison_model
    alerts, poison_points, iterations, poisoned_model_file_name, _ = poison_func(conf, model_name, start_poison, x_att,
                                                                                 x_train)

    print("Test configuration", conf, "\nResults: ", alerts,
          " alerts", poison_points, "poison_points", iterations, "iterations")

    time_taken = time.time() - start_time
    print("Took ", time_taken)
    return alerts, poison_points, iterations, time_taken


def syn_swat_test(att_num=None):
    X, Y, attacks = read_swat()
    syn_attacks = dict()
    syn_attacks[3] = dict(
        field_names=['LIT101', "FIT101", "MV101", "P101"],  # sensor_cols#
        extended_att_len=1100,  # total length of the attack input, taken from the length of attack 33
        att_len=77,
        att_start=700,
        att_magn=0.33,
        att_point="CUSTOM_LINEAR",
        period_start=200
    )

    syn_attacks[33] = dict(  # in reality, "FIT101" and "MV101" are stopped as the tank seems full
        field_names=['LIT101', "FIT101", "MV101", "P101"],  # sensor_cols#
        extended_att_len=900,  # total length of the attack input, taken from the length of attack 33
        att_len=88,
        att_start=450,
        att_magn=1,
        att_point="CUSTOM_FIXED",
        period_start=0
    )

    syn_attacks[36] = dict(
        field_names=['LIT101', "FIT101", "MV101", "P101"],  # sensor_cols#
        extended_att_len=1400,  # total length of the attack input, taken from the length of attack 33
        att_len=106,
        att_start=260,
        att_magn=-1,
        att_point="CUSTOM_FIXED",
        period_start=0
    )

    syn_attacks[31] = dict(
        field_names=["LIT401", 'P302', 'LIT301', "P402"],
        extended_att_len=1100,  # total length of the attack input, taken from the length of attack 33
        att_len=76,
        att_start=800,
        period_start=100,
        att_magn=-1,
        att_point="CUSTOM_FIXED"
    )

    syn_attacks[7] = dict(  # in reality FIT201 is stopped as the tank seems full
        field_names=['LIT301', "FIT201", 'P302'],  # sensor_cols#
        extended_att_len=1400,  # total length of the attack input, taken from the length of attack 33
        att_len=86,
        att_start=475,
        att_magn=1.0,
        att_point="CUSTOM_FIXED",
        period_start=500
    )

    # att 16
    syn_attacks[16] = dict(  # in reality, FIT201 is stopped as th tank seems full
        field_names=['LIT301', "FIT201", 'P302'],  # sensor_cols#
        extended_att_len=1400,  # total length of the attack input, taken from the length of attack 33
        att_len=56,
        att_start=536,
        att_magn=-0.5,
        att_point="CUSTOM_LINEAR",
        period_start=500
    )

    # att 32
    syn_attacks[32] = dict(  # in reality, FIT201 is started as the tank seems empty
        field_names=['LIT301', "FIT201", 'P302'],  # sensor_cols#
        extended_att_len=1200,  # total length of the attack input, taken from the length of attack 33
        att_len=120,
        att_start=570,
        att_magn=1,
        att_point="CUSTOM_FIXED",
        period_start=300
    )

    conf = dict(
        adv_lr=0.2,
        seq_len=2,
        total_len=50,
        code_ratio=1.5,
        signal_func=np.sin,  # signal.square,#signal.sawtooth#np.sin
        att_magn=0.4,
        att_point="CUSTOM_FIXED",
        att_len=1,
        window=10,
        threshold=0.2,
        layers=1,  # increasing it did not show any significant changes
        inflate_factor=2,
        silent=False,
        batches=10,
        it_count=10,
        adv_it_count=40,
        lr=0.6,
        train_points=10,
        max_adv_iter=300,
        periods=5,
        randomize=True,
        optimizer=tf.train.GradientDescentOptimizer(0.6),
        # tf.train.MomentumOptimizer(0.6, 0.5)#tf.train.AdamOptimizer(0.05) #c.lr
        generator=sin_generator_c(),
        single_sequence=False,
        naive=True,
        find_poison=False,
        retrain=False,
        retrain_points=10,
        max_clean_poison=100,
        activation=tf.nn.tanh,
        activate_last=True,
        partial_attacked=[0],
        predict_one=False,
        bo_max_norm=False,
    )
    conf["silent"] = False
    conf["it_count"] = 10
    conf["seq_len"] = 1
    cols = list(conf.keys()) + ['alerts', 'poison_points', 'iterations', 'time']
    df_results = pd.DataFrame(columns=cols)
    if att_num is None:
        attacks_list = list(syn_attacks.keys())
    else:
        attacks_list = [att_num]
    for at in attacks_list:
        print("Attack", at)
        field_names = syn_attacks[at]["field_names"]
        extended_att_len = syn_attacks[at]["extended_att_len"]
        conf["att_len"] = syn_attacks[at]["att_len"]
        conf["att_start"] = syn_attacks[at]["att_start"]
        conf["att_magn"] = syn_attacks[at]["att_magn"]
        conf["att_point"] = syn_attacks[at]["att_point"]
        conf["period_start"] = syn_attacks[at]["period_start"] if "period_start" in syn_attacks[at] else 0
        conf["field_names"] = field_names

        if at in [3, 16, 31]:
            conf["threshold"] = 0.1
        else:
            conf["threshold"] = 0.2

        x_train_full = X[field_names].values

        for conf["threshold"] in np.arange(0.06, 0.2, 0.02):
            for conf["seq_len"] in [2, 10, 20, 30]:
                failed = 0
                for _ in range(5):
                    conf["train_points"] = 5
                    conf["total_len"] = conf["train_points"] * extended_att_len
                    res = conf.copy()
                    res['alerts'], res['poison_points'], res['iterations'], res['time'] = run_synthetic_swat_test(conf,
                                                                                                                  x_train_full,
                                                                                                                  extended_att_len)
                    df_results = df_results.append(pd.DataFrame([res]), ignore_index=True)
                    # add to the test results!
                    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                    df_results.to_csv(data_path + "PoisoningTests_swat_syn_%d_%s.csv" % (at, timestamp))
                    if res['alerts']:
                        failed += 1
                        if failed > 2:
                            break

    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    df_results.to_csv(data_path + "PoisoningTests_swat_syn_%s.csv" % (timestamp))


def run_test(conf, x_train_full, x_att):
    x_train = x_train_full[-conf["total_len"]:]
    print(x_train_full.shape, x_train.shape, x_att.shape)
    # it will be used for generating a clean validation input
    conf["generator"] = echo_generator_c(x_train_full[:len(x_att)])

    start_time = time.time()
    model_name = "./tmp/SWaT_seq_%d" % (conf["seq_len"])
    model_signal(conf, model_name, x_train_full)

    # moderate the attack
    x_att = np.clip(x_att, -conf["att_magn"], 1 + conf["att_magn"])

    org_silent = conf["silent"]
    #     conf["silent"] = True
    start_poison, alerts = find_start_poison(conf, model_name, x_att, x_train)
    conf["silent"] = org_silent

    if not alerts:
        poison_func = poison_model_naive if conf["naive"] else poison_model
        alerts, poison_points, iterations, poisoned_model_file_name = poison_func(conf, model_name, start_poison, x_att,
                                                                                  x_train)

        print("Test configuration", conf, "\nResults: ", alerts,
              " alerts", poison_points, "poison_points", iterations, "iterations")
    else:
        print("Got ", alerts, " alerts with closest poison, not continuing")
        poison_points, iterations = 0, 0

    time_taken = time.time() - start_time
    print("Took ", time_taken)
    return alerts, poison_points, iterations, time_taken


def swat_test():
    X, Y, attacks = read_swat()
    conf = dict(
        adv_lr=0.05,
        seq_len=2,
        total_len=50,
        code_ratio=2,
        signal_func=np.sin,  # signal.square,#signal.sawtooth#np.sin
        att_magn=0.5,
        att_point="SIN_BOTTOM",
        att_len=1,
        window=10,
        threshold=0.2,
        layers=1,  # increasing it did not show any significant changes
        inflate_factor=2,
        silent=True,
        batches=1,
        it_count=300,
        adv_it_count=40,
        lr=0.6,
        train_points=10,
        max_adv_iter=1000,
        periods=5,
        randomize=False,
        optimizer=tf.train.GradientDescentOptimizer(0.6),
        # tf.train.MomentumOptimizer(0.6, 0.5)#tf.train.AdamOptimizer(0.05) #c.lr
        generator=sin_generator_c(),
        single_sequence=False,
        naive=False,
        find_poison=False,
        retrain=False,
        retrain_points=10,
        max_clean_poison=100,
        activation=tf.nn.tanh,
        activate_last=False
    )

    # conf["partial"] = True #TODO - test when the poison can change only 1 signal (how)
    # conf["partial_attacked"] = [0]

    cols = list(conf.keys()) + ['alerts', 'poison_points', 'iterations', 'time']

    conf["it_count"] = 10
    conf["total_len"] = conf["train_points"] * len(x_att)
    conf["silent"] = False
    conf["naive"] = True
    conf["att_magn"] = 0.3
    conf["batches"] = 10
    conf["randomize"] = False
    conf["threshold"] = 0.2  # for attack 31
    conf["optimizer"] = tf.train.GradientDescentOptimizer(conf["lr"])

    for at in [7]:
        att_start = datetime.datetime.strptime(attacks[at]["attack_start"], time_format)
        att_end = datetime.datetime.strptime(attacks[at]["attack_end"], time_format)
        field_names = attacks[at]["field_names"]
        delta_hours = 3
        if "delta_hours" in attacks[at]:
            delta_hours = attacks[at]["delta_hours"]
        delta = datetime.timedelta(hours=delta_hours)
        x_att = Y[att_start - delta:att_end + delta][field_names].values
        x_train_full = X[field_names].values

        df_results = pd.DataFrame(columns=cols)
        for conf["naive"] in [True, False]:
            for conf["att_magn"] in [0.35]:  # np.arange(0.3, 1.0, 0.1):
                for conf["train_points"] in [10, 20]:
                    conf["total_len"] = conf["train_points"] * len(x_att)
                    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                    df_results.to_csv(data_path + "PoisoningTests_attack_%d_%s.csv" % (at, timestamp))
                    # add to the test results!
                    res = conf.copy()
                    # should I pass all the data for training?
                    res['alerts'], res['poison_points'], res['iterations'], res['time'] = run_test(conf, x_train_full,
                                                                                                   x_att)
                    df_results = df_results.append(pd.DataFrame([res]), ignore_index=True)
                    if res['alerts']:
                        break

        timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        df_results.to_csv(data_path + "PoisoningTests_attack_%d_%s.csv" % (at, timestamp))


# # Predictor!
# It looks I have to consider the model's reaction on poison
# And for that I need a predictor!
# For the fun of it, using Keras
#

def create_model(inp_seq_len, features, out_features, out_seq_len, activation, layers, bootstrap_shape=None):
    input_layer = Input(shape=(inp_seq_len, features), dtype='float32') 
    if bootstrap_shape is not None:
        input_bootstr_layer = Input(shape=bootstrap_shape)
        layer = Concatenate(axis=-1)([Flatten(name="flat")(input_layer), Flatten(name="bootstr_flat")(input_bootstr_layer)])
    else:    
        layer = Flatten(name="flat")(input_layer)
    for l_idx, l_size in enumerate(layers):
        layer = Dense(l_size, activation=activation, name="dense_%d" % l_idx)(layer)
    layer = Dropout(.2)(layer)
    last_layer = Dense(out_seq_len * out_features, name="dense_last", activation='tanh')(layer)
    output_layer = Reshape((out_seq_len, out_features))(last_layer)
    if bootstrap_shape is not None:
        return Model(inputs=[input_layer, input_bootstr_layer], outputs=output_layer)
    else:    
        return Model(inputs=input_layer, outputs=output_layer)



def create_conv_model(inp_seq_len, features, out_features, out_seq_len, activation, layers, kernel_size=2, filters=16):
    input_layer = Input(shape=(inp_seq_len, features), dtype='float32')
    layer = input_layer
    for l_idx in range(layers):
        if layer.shape[-2] <= kernel_size:
            kernel_size = 1
        print(layer.shape, kernel_size)
        layer = Conv1D(filters=filters, kernel_size=kernel_size,
                       activation=activation, name="conv_%d" % l_idx)(layer)
        layer = MaxPooling1D()(layer)
        #         if l_idx%2:
        filters *= 2

    layer = Flatten(name="flat_last")(layer)
    #     layer = Dropout(.2)(layer)
    last_layer = Dense(out_seq_len * out_features, name="dense_last")(layer)
    output_layer = Reshape((out_seq_len, out_features))(last_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


# prepare sliced inputs
def get_ae_sequences(X, _, seq_len, out_seq_len=1, lag=0):
    print("get_ae_sequences")
    return [X[start:start + seq_len, :] for start in np.arange(len(X) - seq_len - out_seq_len - lag)], [
        X[start + seq_len:start + seq_len + out_seq_len, :] for start in np.arange(lag, len(X) - seq_len - out_seq_len)]


def get_cross_pred_sequences(X, Y, seq_len, out_seq_len=1, lag=0):
    # print("!!!!!!!!!!!!get_cross_pred_sequences!!!!!!!!!!!!!!")
    # print("get_cross_pred_sequences", type(X), X.shape, type(Y), Y.shape, seq_len, out_seq_len, lag)
    return [X[start:start + seq_len, :] for start in np.arange(len(X) - seq_len + 1 - lag)], [
        Y[start:start + out_seq_len, :] for start in np.arange(lag, len(Y) - seq_len + 1)]


def create_and_train_predictor(X, field_names, seq_len, out_seq_len=1,
                               horizon=0,
                               data_generator=get_cross_pred_sequences, silent=False,
                               out_field_names=None, layers=None, conv=False, conv_filters=16, bootstrX=None):
    if conv:
        model = create_conv_model(seq_len, len(field_names), len(field_names), out_seq_len=1,
                                  activation="relu", layers=layers, kernel_size=2, filters=conv_filters)
    else:
        if layers is None:
            inp_len = seq_len
            if bootstrX is not None:
                inp_len += bootstrX.shape[-2]
            layers = [inp_len * len(field_names) * 2,
                      int(inp_len * len(field_names) / 2),
                      inp_len * len(field_names) * 2]
        model = create_model(seq_len, len(field_names),
                             len(out_field_names) if out_field_names is not None else len(field_names), out_seq_len,
                             "relu",
                             layers, bootstrX.shape if bootstrX is not None else None)

    model.compile(loss='mean_squared_error', optimizer='adam')
    if not silent:
        model.summary()

    Y = X[out_field_names].values if out_field_names is not None else None
    sliced_X, sliced_Y = data_generator(X[field_names].values, Y, seq_len, out_seq_len, lag=horizon)

    # split into training and validation
    X1, X2, Y1, Y2 = train_test_split(sliced_X, sliced_Y, test_size=0.33, random_state=42)

    # train models with early stopping and reduction of learning rate on plateau
    # TODO - save the best model
    earlyStopping = EarlyStopping(monitor='val_loss', patience=3,
                                  verbose=1 if not silent else 0,
                                  min_delta=1e-4, mode='auto')
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1,
                                   verbose=1 if not silent else 0,
                                   min_delta=1e-4, mode='min')

    # initialize time
    start_time = time.time()

    # train autoencoder
    if bootstrX is not None:
        fitX = [np.array(X1), np.array([bootstrX.values] * len(X1))]
        valX = [np.array(X2), np.array([bootstrX.values] * len(X2))]
    else:
        fitX = np.array(X1)
        valX = np.array(X2)
    history = model.fit(fitX, np.array(Y1),
                        epochs=500,
                        batch_size=32,
                        shuffle=True,
                        callbacks=[earlyStopping, lr_reduced],
                        verbose=2 if not silent else 0,
                        validation_data=(valX, np.array(Y2)))

    # model_file_name = "./tmp/te_seq_%d_out_seq%d.h5" % (seq_len, out_seq_len)
    # model.save(model_file_name)

    if False:  # not silent:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    return model


def load_cross_predict_np(model_file_name, Y, in_mask, out_mask, seq_len, out_seq_len, lag=0, bootstrX=None):
    # print("==> load_cross_predict_np", in_mask, out_mask)
    # Test on the attack input
    X_att = Y[:, in_mask]
    Y_att = Y[:, out_mask]

    sliced_X_att, sliced_Y_att = get_cross_pred_sequences(X_att, Y_att,
                                                          seq_len, out_seq_len,
                                                          lag=lag)  # get_sequences(X_att.values, seq_len)

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    #     # otherwise, previous default graph resets confuse Keras completely
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
        with tf.Session(graph=g, config=config).as_default():
            model = load_model(model_file_name)
            if bootstrX is None:
                Y_att_hat = model.predict(np.array(sliced_X_att))
            else:
                Y_att_hat = model.predict([np.array(sliced_X_att), np.array([bootstrX] * len(sliced_X_att))])

    Y_att = np.concatenate(sliced_Y_att)
    Y_att_hat = Y_att_hat.reshape((-1, Y_att_hat.shape[-1]))
    
    # the length of the output is shorter than the input, thus appending the original first values
    # Y_att = np.concatenate([Y_att[:seq_len-1] , Y_att])
    # Y_att_hat = np.concatenate([Y_att[:seq_len-1] , Y_att_hat])
    # print("<== load_cross_predict_np, length(Y), length(Y_att), length(Y_att_hat)", len(Y), len(Y_att), len(Y_att_hat))
    return Y_att, Y_att_hat

# We need to train the process predictor on multiple simulation runs
# prior to using it!
def train_process_predictor(att_idx):
    # read a matlab simulations with attacks of different magnitude
    # iterate through files
    test_name = "TE_attack_%d"%att_idx
    sim_data_csv_name = "TE_attack_%d"%att_idx + "_sim_poison.csv"
    if not os.path.exists(sim_data_csv_name):
        print("Creating the dataset")
        df_all = None
        i = 0
        for f in glob.glob("." + os.sep + "\TESimData_att%d"%att_idx + os.sep + test_name + '*_xmeas.mat'):
            print(f)
            te_base_name = f[:f.find("_xmeas.mat")]
            df = te_mat_to_csv(te_base_name, save=False)
            if df_all is None:
                df_all = df
            else:
                df_all = df_all.append(df)
        df_all.to_csv(sim_data_csv_name, index=None)
    else:
        print("Loading the dataset")
        df_all = pd.read_csv(sim_data_csv_name)

    scaler = MinMaxScaler((-1, 1))
    scaler.fit(df_all)
    scaler_file_name = "./tmp/te_attack_%d_scaler.pkl"%(att_idx)
    pickle.dump(scaler, open(scaler_file_name, 'wb'))
    X = pd.DataFrame(index = df_all.index, data = scaler.transform(df_all),columns=df_all.columns)
    del df_all

    att_test_name = "TE_attack_%d_with_poison.csv"%att_idx

    # Make sure to use the correctly calibrated scaler!
    df_te_attack_test = pd.read_csv(att_test_name)
    df_te_attack_test = pd.DataFrame(index = df_te_attack_test.index, 
                                    data = scaler.transform(df_te_attack_test),
                                    columns=df_te_attack_test.columns)
        
    if att_idx == 1:
        in_field_names = ["xmeas_1"] 
        out_field_names = ["xmeas_7", "xmeas_8", "xmv_3"]
        bootstr_fields = ["xmeas_1", "xmeas_7", "xmeas_8", "xmv_3"]
    elif att_idx == 2 or att_idx == 3:
        in_field_names = ["xmeas_14"] 
        out_field_names = ["xmv_7", "xmeas_15", "xmv_8"]
        bootstr_fields = ["xmeas_14", "xmv_7", "xmeas_15", "xmv_8"]
    elif att_idx == 31:
        in_field_names = ["xmeas_14"] 
        out_field_names = ["xmv_7", "xmeas_12"]
        bootstr_fields = ["xmeas_14", "xmv_7", "xmeas_12"]
    horizon = 0
    out_seq_len=1

    df_res = pd.DataFrame(columns=["pred_seq_len", "bootstraplen", "squared_error", "alerts"])

    for pred_seq_len in [100, 200, 300, 400]:
        for bootstraplen in [50]: 
            for _ in range(1):
                bootstrX=None if bootstraplen ==0 else df_te_attack_test[bootstr_fields][:bootstraplen]
                model_file_name = "./tmp/te_attack_%d_pred_seq_%d_bs_%d.h5"%(att_idx, pred_seq_len, bootstraplen)
                scaler_file_name = "./tmp/te_attack_%d_scaler.pkl"%(att_idx)

                config = tf.ConfigProto(log_device_placement=False)
                config.gpu_options.allow_growth = True
                tf.reset_default_graph()
                with tf.Session(config=config).as_default():
                    if False:#os.path.exists(model_file_name):
                        print("Loading existing model", model_file_name)
                        model = load_model(model_file_name)
                        scaler = load(open(scaler_file_name, 'rb'))
                    else:
                        model = create_and_train_predictor(X[bootstraplen:], in_field_names, pred_seq_len, out_seq_len,
                                                                horizon,
                                                                data_generator=get_cross_pred_sequences,
                                                                out_field_names=out_field_names, bootstrX=bootstrX)
                        model.save(model_file_name)
                    
                    sliced_X, sliced_Y = get_cross_pred_sequences(df_te_attack_test[in_field_names].values,
                                                                df_te_attack_test[out_field_names].values,
                                                                pred_seq_len, out_seq_len, lag=horizon)


                    if bootstrX is not None:
                        Y_hat = model.predict([np.array(sliced_X), np.array([bootstrX.values] * len(sliced_X))])
                    else:
                        Y_hat = model.predict(np.array(sliced_X))
                        
                    del model
                    
                    Y_true = np.concatenate(sliced_Y)
                    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
                    print("lengths", len(df_te_attack_test), len(Y_hat), len(Y_true))
                    
                    df_y = pd.DataFrame(Y_true, columns=out_field_names)
                    df_y_hat = pd.DataFrame(Y_hat, columns=out_field_names)

                    df_y.plot(title="Real")
                    df_y_hat.plot(title="Prediction")
                    (df_y-df_y_hat).plot(title="Residual")
                    ((df_y-df_y_hat).abs().rolling(conf["window"]).min()>conf["threshold"]).astype(int).plot(title="Alert")
                    alerts =((df_y-df_y_hat).abs().rolling(conf["window"]).min()>conf["threshold"]).astype(int).sum() 
                    squared_error = (df_y - df_y_hat).pow(2).sum()
                    print("pred_seq_len ", pred_seq_len, "\nsquared_error\n", squared_error,"\nalerts\n",alerts)
                    df_res = df_res.append({
                        "pred_seq_len": pred_seq_len,
                        "bootstraplen": bootstraplen,
                        "squared_error":str(squared_error),
                        "alerts":str(alerts)
                    }, ignore_index=True)
    return df_res

# # TE tests
#

# Reading and converting the matlab files
def te_mat_to_csv(te_base_name):
    mat = scipy.io.loadmat(te_base_name + '_xmeas.mat')
    xmeas = mat["simout"]
    mat = scipy.io.loadmat(te_base_name + '_xmvs.mat')
    xmvs = mat["xmv"]
    x_train = np.concatenate([xmeas, xmvs], axis=1)
    xmeas_cols = ["xmeas_%d" % i for i in range(1, xmeas.shape[1] + 1)]
    xmv_cols = ["xmv_%d" % i for i in range(1, xmvs.shape[1] + 1)]
    df_train = pd.DataFrame(x_train, columns=xmeas_cols + xmv_cols)
    df_train.to_csv(te_base_name + ".csv", index=None)
    return df_train


def read_te_data(te_base_name, scale_to_half=True):
    # TODO - a dict with attacks details
    att_idx = int(te_base_name.split("_")[-1])
    print("Reading data for attack %d, base name %s" % (att_idx, te_base_name))

    df_te_train_clean = pd.read_csv("TE_train.csv")
    df_te_train = pd.read_csv("TE_train_for_attack%d.csv" % att_idx)
    # df_te_train = pd.read_csv("TE_train_for_attack2.csv")
    te_att_csv_file = te_base_name + ".csv"
    te_att_mat_file = te_base_name + '_xmeas.mat'
    if not os.path.exists(te_att_csv_file) or os.path.getmtime(te_att_mat_file) > os.path.getmtime(te_att_csv_file):
        print("Converting", te_att_mat_file)
        df_te_attack = te_mat_to_csv(te_base_name)
    else:
        df_te_attack = pd.read_csv(te_att_csv_file)

    sig_period = 705  # determined by autocorr
    # round the attack to the signal's period
    # we better start poisoning at the same location relative to the period
    # leaving some time after the attack is necessary for the system to stabilize.

    round_att_len = len(df_te_attack) // sig_period * sig_period
    df_te_attack = df_te_attack[:round_att_len]
    combined_data = df_te_train.append(df_te_attack, ignore_index=True)

    # scale sensor data
    if scale_to_half:  # synthetic data
        scaler = MinMaxScaler((-0.5, 0.5))
    else:  # real data
        scaler_file_name = "./tmp/te_attack_%d_scaler.pkl"%att_idx
        if os.path.exists(scaler_file_name):
            print("Loading existing scaler", scaler_file_name)
            scaler = pickle.load(open(scaler_file_name, 'rb'))
        else:
            scaler = MinMaxScaler((-1, 1))
            scaler.fit(combined_data)

    Y = pd.DataFrame(index=df_te_attack.index, data=scaler.transform(df_te_attack), columns=df_te_attack.columns)
    X = pd.DataFrame(index=df_te_train.index, data=scaler.transform(df_te_train), columns=df_te_train.columns)
    X_clean = pd.DataFrame(index=df_te_train_clean.index, data=scaler.transform(df_te_train_clean),
                           columns=df_te_train_clean.columns)
    # df_te_attack["xmeas_14"].plot()
    return X, Y, scaler, X_clean


def convert_2_matlab(poison_pkl_file, mat_name, field_names, scaler, all_columns):
    with open(poison_pkl_file, "rb") as f:
        good_poison = pickle.load(f)

    upsampled_arrays = []
    for p_idx, p in enumerate(good_poison):  # shape (1, 2001, 3)
        print("Poison %d" % p_idx)
        upsampled = np.empty((p.shape[1] * 100, p.shape[2]))
        print(p.shape, upsampled.shape)
        for col in range(p.shape[2]):
            s = interpolate.InterpolatedUnivariateSpline(np.arange(p.shape[1]) * 100, p[0, :, col])
            upsampled[:, col] = s(np.arange(len(upsampled)))
        poisons_dict = dict()
        dummy = np.zeros((len(upsampled), len(all_columns)))
        for f_idx, f in enumerate(field_names):
            dummy[:, all_columns.get_loc(f)] = upsampled[:, f_idx]

        upsampled = scaler.inverse_transform(dummy)
        upsampled_arrays.append(upsampled)
        # append another empty data to create padding between the real poison
        # dummy = np.zeros((len(upsampled), len(all_columns)))
        # upsampled_arrays.append(dummy)

        for f_idx, f in enumerate(field_names):
            poisons_dict["p_xmeas_%d" % all_columns.get_loc(f)] = upsampled[:, all_columns.get_loc(f)]

        print("Saving poison %d" % p_idx, "to", '%s%d.mat' % (mat_name, p_idx))
        scipy.io.savemat('%s%d.mat' % (mat_name, p_idx), poisons_dict)

    combined_poison = np.concatenate(upsampled_arrays)
    poisons_dict = dict()
    for f_idx, f in enumerate(field_names):
        poisons_dict["p_xmeas_%d" % all_columns.get_loc(f)] = combined_poison[:, all_columns.get_loc(f)]
    print("Saving combined poison", "to", '%s.mat' % mat_name)
    scipy.io.savemat(mat_name + '.mat', poisons_dict)


def test_model(conf, model_name, it_count, x_train, poisons, x_test):
    c = munchify(conf)

    train_err, test_err, n_out, alerts, _, max_err, _ = train_and_test_adv(model_name, 0, it_count, x_train,
                                                                           poisons, poisons[-1],
                                                                           x_test, c, False, 
                                                                           not c.silent,
                                                                           title="Real poison test")
    print("Got %d alerts, max_err %f" % (alerts, max_err))
    return alerts, max_err


def test_model_on_attack(conf, x_train_full, extended_att_len, x_att, start_poison):
    print("Testing conf: ", repr(conf))
    start_time = time.time()
    model_name = "./tmp/te_seq_%d" % (conf["seq_len"])
    # remove old intermediate models
    for f in glob.glob("." + os.sep + "tmp" + os.sep + model_name + '_*.*'):
        os.remove(f)

    np.random.seed(1)
    conf["total_len"] = conf["train_points"] * extended_att_len
    x_train = x_train_full[-conf["total_len"]:]
    period_start = conf["period_start"] if "period_start" in conf else 0
    # it will be used for generating a clean validation input
    conf["generator"] = echo_generator_c(x_train_full[period_start:extended_att_len])

    _, conf["adv_it_count"] = model_signal(conf, model_name, x_train_full)

    if conf.get("att_start", None) is not None:
        _, x_att = generate_poison_and_attack(conf)

    alerts, max_error = test_model(conf, model_name, 0, x_train, [start_poison], x_att)
    print("Result: alerts %d, max_error %f" % (alerts, max_error))
    return alerts, max_error


def test_te_attack(conf, model_name, mat_name, poison_name, scaler, x_train, field_names):
    df_te_poison_attack = te_mat_to_csv(conf["te_base_name"] + "_with_poison")
    Y_poison_attack = pd.DataFrame(index=df_te_poison_attack.index,
                                   data=scaler.transform(df_te_poison_attack),
                                   columns=df_te_poison_attack.columns)

    with open(poison_name, "rb") as f:
        good_poison = pickle.load(f)

    train_len = np.sum([len(p[0]) for p in good_poison])

    poisons = []
    offset = 0
    for p_idx in range(len(good_poison)):
        x_poison_ = Y_poison_attack[field_names].values[offset:offset + len(good_poison[0][0])]
        poisons.append(x_poison_[np.newaxis, :, :])
        offset += len(good_poison[0][0])

    # x_poison_2 = Y_poison_attack[field_names].values[len(x_poison_1):len(x_poison_1)+len(good_poison[2][0])]
    x_att = Y_poison_attack[field_names].values[train_len:]
    print('x_att', x_att.shape, "train_len", train_len, len(df_te_poison_attack))

    # Let's we how well we cope with REAL poison
    # 1. Attack with no poisoning:
    conf["silent"] = False
    org_alerts, _ = test_model(conf, model_name, 0, x_train, poisons[0], x_att)
    # # 2. Poison 0 (no prior poisoning)
    # test_model(conf, model_name, 0, x_train, x_poison_0, x_poison_0)
    # # Poison 1 (no prior poisoning)
    # test_model(conf, model_name, 0, x_train, x_poison_0, x_poison_1)
    # # Poison 1 (with poisoning)
    # test_model(conf, model_name, conf["it_count"], x_train, x_poison_0, x_poison_1)
    # # Poison 2 (with poisoning)
    # test_model(conf, model_name, conf["it_count"], x_train, np.concatenate([x_poison_0,x_poison_1]), x_poison_2)
    # Attack with poisoning
    alerts, _ = test_model(conf, model_name, conf["it_count"], x_train, poisons, x_att)
    return org_alerts, alerts


def run_te_test(conf, x_train_full, extended_att_len, x_att, start_poison, field_names, scaler, all_columns, test_model_only):
    start_time = time.time()
    model_name = "./tmp/te_attack_%d_seq_%d" % (conf["att_num"], conf["seq_len"])
    # remove old intermediate models
    for f in glob.glob("." + os.sep + "tmp" + os.sep + model_name + '_*.*'):
        os.remove(f)

    np.random.seed(1)
    conf["total_len"] = conf["train_points"] * extended_att_len
    x_train = x_train_full[-conf["total_len"]:]
    period_start = conf["period_start"] if "period_start" in conf else 0
    # it will be used for generating a clean validation input
    conf["generator"] = echo_generator_c(x_train_full[period_start:extended_att_len])

    # reset the iterations counter to the one actually used
    if not test_model_only:
        model_signal(conf, model_name, x_train_full)

    if conf.get("att_start", None) is not None:
        _, x_att = generate_poison_and_attack(conf)

    poison_func = poison_model_naive if conf["naive"] else poison_model
    alerts = poison_points = iterations = 0

    if not test_model_only:
        print("Testing configuration", conf)

        alerts, poison_points, iterations, _, poisons = poison_func(conf, model_name, start_poison,
                                                                                           x_att, x_train)
        print("Test configuration", conf, "\nResults: ", alerts,
              " alerts", poison_points, "poison_points", iterations, "iterations")

    if alerts == 0 and conf.get("sim", True):
        poison_name = "te_attack_%d_seq_%d_poison.pkl" % (conf["att_num"], conf["seq_len"])
        mat_name = conf["te_base_name"].lower() + "_poison"
        if not test_model_only:
            print("Saving successful poison to ", poison_name)
            with open(poison_name, "wb") as f:
                pickle.dump(poisons, f)
            convert_2_matlab(poison_name, mat_name, field_names, scaler, all_columns)
            # Run Matlab
            os.chdir("./temexd_mod")
            subprocess.run(["matlab", "-nosplash", "-nodesktop", "-batch",
                            "\"n_poisons=%d;att_idx=%d;run('convert_poison.m');\"" % (poison_points, conf["att_num"])])
            os.chdir("..")
        # Test it!
        org_alerts, final_alerts = test_te_attack(conf, model_name, mat_name, poison_name, scaler, x_train, field_names)
        if final_alerts:
            print("FAIL!", final_alerts)
        else:
            print("SUCCESS!")

    time_taken = time.time() - start_time
    print("Took ", time_taken)
    return org_alerts, alerts, final_alerts, poison_points, iterations, time_taken


def te_test(test_model_only = False):
    att_start = 100
    att_len = 200
    att_magn = -6.5  # unscaled

    conf = dict(
        adv_lr=0.2,
        seq_len=20,
        total_len=50,
        code_ratio=1.5,
        signal_func=np.sin,  # signal.square,#signal.sawtooth#np.sin
        att_magn=0.4,
        att_point="CUSTOM_FIXED",
        att_len=1,
        window=10,
        threshold=0.2,
        layers=1,  # increasing it did not show any significant changes
        inflate_factor=2,
        silent=False,
        batches=10,
        it_count=30,
        adv_it_count=20,
        lr=0.6,
        train_points=10,
        max_adv_iter=300,
        periods=5,
        randomize=True,
        optimizer=tf.train.GradientDescentOptimizer(0.6),
        # tf.train.MomentumOptimizer(0.6, 0.5)#tf.train.AdamOptimizer(0.05) #c.lr
        generator=sin_generator_c(),
        single_sequence=False,
        naive=False,
        find_poison=True,
        retrain=False,
        retrain_points=10,
        max_clean_poison=100,
        activation=tf.nn.tanh,
        activate_last=True,
        partial_attacked=[0],
        pred_seq_len = 400, 
        pred_bs_len = 50,
        batch_norm=False,
        min_delta=0.0001,
        patience=30,
        sim=True,
        # dropout_rate=0.5 - to keep it simple, as the back gradient with dropout might be tricky
        # field_names = ["xmeas_14", "xmv_7", "xmeas_15", "xmv_8"]
        # For attack 31 we should use ["xmeas_14", "xmv_7", "xmeas_12"]
        field_names = ["xmeas_14", "xmv_7", "xmeas_12"]        
    )
    conf["silent"] = False
    conf["it_count"] = conf["adv_it_count"] = 30

    if not conf.get("sim", False):
        conf["seq_len"] = 22
        conf["train_points"] = 5
        conf["inflate_factor"] = 0
        conf["code_ratio"] = conf["seq_len"]/1.5
        conf["patience"] = 30
        conf["partial_attacked"]=[0]

    att_num = None

    cols = list(conf.keys()) + ['org_alerts', 'alerts', 'final_alerts', 'poison_points', 'iterations', 'time']
    df_results = pd.DataFrame(columns=cols)


    for conf["att_num"] in [1]:
        if conf["att_num"] == 1:
            conf["field_names"] = ["xmeas_1", "xmeas_7", "xmeas_8", "xmv_3"]
        if conf["att_num"] == 2 or conf["att_num"] == 3:
            conf["field_names"] = ["xmeas_14", "xmv_7", "xmeas_15", "xmv_8"]
        elif conf["att_num"] == 31:
            conf["field_names"] = ["xmeas_14", "xmv_7", "xmeas_12"]

        if not conf.get("sim", False):  # synthetic attacks
            if conf["att_num"] == 2:
                conf["att_len"] = 260
                conf["att_start"] = 705
                # considering scaling between 0.5 and -0.5, the real attack magnitude of 7
                # and real signal magnitude of ~26 the value should be 0.3 - 0.13, but this is below the threshold
                conf["att_magn"] = 0.05
                conf["att_point"] = "CUSTOM_FIXED"
                att_magn_range = [0.3]  # [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            elif conf["att_num"] == 3:
                conf["att_len"] = 260
                conf["att_start"] = 705
                # considering scaling between 0.5 and -0.5, the real attack magnitude of 7
                # and real signal magnitude of ~26 the value should be 0.3 - 0.13, but this is below the threshold
                conf["att_magn"] = 0
                conf["att_point"] = "CUSTOM_FIXED"
                att_magn_range = [-0.3]  # np.arange(0, -0.6, -0.1)

        # IMPORTANT: attack idx should be the last one and preceded by _
        te_base_name = "TE_attack_%d" % conf["att_num"]
        X, Y, scaler, X_clean = read_te_data(te_base_name, scale_to_half=not conf.get("sim", False))
        extended_att_len = len(Y)
        conf["te_base_name"] = te_base_name
        field_names = conf["field_names"]

        x_train_full = X[field_names].values
        x_att = Y[field_names].values

        # Y_real is way too different
        start_poison = X_clean[field_names].values[:len(x_att)]
        print(len(x_att))
        if "pred_seq_len" in conf:
            out_seq_len = 1
            horizon = 0
            out_mask = list(set(np.arange(len(field_names))) - set(conf["partial_attacked"]))
            in_field_names = np.array(field_names)[conf["partial_attacked"]].tolist()
            out_field_names = np.array(field_names)[out_mask].tolist()
#            dummy_model = create_and_train_predictor(X, in_field_names, conf["pred_seq_len"], out_seq_len,
#                                                     horizon,
#                                                     data_generator=get_cross_pred_sequences,
#                                                     out_field_names=out_field_names)
#            del dummy_model
        for conf["naive"] in [True]:
            for conf["att_magn"] in [0.05]: #[0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
                failed = 0
                test_cnt = 1
                for _ in range(test_cnt):
                    conf["total_len"] = conf["train_points"] * extended_att_len
                    res = conf.copy()
                    res['org_alerts'], res['alerts'], res['final_alerts'],  res['poison_points'], res['iterations'], res[
                        'time'] = run_te_test(conf, x_train_full,
                                              extended_att_len,
                                              x_att, start_poison,
                                              field_names, scaler,
                                              X.columns, test_model_only)

                    df_results = df_results.append(pd.DataFrame([res]), ignore_index=True)
                    # add to the test results!
                    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                    df_results.to_csv(data_path + "PoisoningTests_te_%s.csv" % timestamp)
                    if not test_model_only and res['alerts']:
                        failed += 1
                        if failed > test_cnt // 2:
                            break


if __name__ == '__main__':
    # synthetic_test()
    # if len(sys.argv) > 1:
    #     syn_swat_test(int(sys.argv[1]))
    # else:
    #     syn_swat_test()
    te_test("-t" in sys.argv)