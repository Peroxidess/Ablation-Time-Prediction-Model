# Copyright (c) 2017 - 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# Runs MNIST experitment. Default 10 runs for 10 random seeds.
#
# Usage:
# python -m mnist.imblanace_mnist_train_ad.py
#
# Flags:
# --exp             [string]         Experiment name, `ours`, `hm`, `ratio`, `random` or `baseline`.
# --pos_ratio       [float]          The ratio for the positive class, choose between 0.9 - 0.995.
# --nrun            [int]            Total number of runs with different random seeds.
# --ntrain          [int]            Number of training examples.
# --nval            [int]            Number of validation examples.
# --ntest           [int]            Number of test examples.
# --tensorboard                      Writes TensorBoard logs while training, default True.
# --notensorboard                    Disable TensorBoard.
# --verbose                          Print training progress, default False.
# --noverbose                        Disable printing.
#
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import os
import six
import tensorflow as tf

from collections import namedtuple
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

from learn_rewieght.reweight import get_model, reweight_random, reweight_autodiff, reweight_hard_mining
from preprocess.load_data import load_data_
from preprocess.get_dataset import get_dataset_, data_preprocessing, anomaly_dectection
from model.training_ import training_model, model_training, precision, recall, f1, r2
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.logging.set_verbosity(tf.logging.ERROR)

flags = tf.flags
flags.DEFINE_float('pos_ratio', 0.995, 'Ratio of positive examples in training')
flags.DEFINE_integer('nrun', 10, 'Number of runs')
flags.DEFINE_integer('ntest', 500, 'Number of testing examples')
flags.DEFINE_integer('ntrain', 5000, 'Number of training examples')
flags.DEFINE_integer('nval', 10, 'Number of validation examples')
flags.DEFINE_bool('verbose', False, 'Whether to print training progress')
flags.DEFINE_bool('tensorboard', False, 'Whether to save training progress')
flags.DEFINE_string('exp', 'baseline', 'Which experiment to run')
FLAGS = tf.flags.FLAGS


Config = namedtuple('Config', [
    'reweight', 'lr', 'num_steps', 'random', 'ratio_weighted', 'nval', 'hard_mining', 'bsize'
])

exp_repo = dict()


def RegisterExp(name):
    def _decorator(f):
        exp_repo[name] = f
        return f

    return _decorator


LR = 0.001
NUM_STEPS = 4000


@RegisterExp('baseline')
def baseline_config():
    return Config(
        reweight=False,
        num_steps=NUM_STEPS * 2,
        lr=LR,
        random=False,
        ratio_weighted=False,
        hard_mining=False,
        bsize=100,
        nval=0)


@RegisterExp('hm')
def baseline_config():
    return Config(
        reweight=False,
        num_steps=NUM_STEPS * 2,
        lr=LR,
        random=False,
        ratio_weighted=False,
        hard_mining=True,
        bsize=500,
        nval=0)


@RegisterExp('ratio')
def ratio_config():
    return Config(
        reweight=False,
        num_steps=NUM_STEPS * 2,
        lr=LR,
        random=False,
        ratio_weighted=True,
        hard_mining=False,
        bsize=100,
        nval=0)


@RegisterExp('random')
def dpfish_config():
    return Config(
        reweight=True,
        num_steps=NUM_STEPS * 2,
        lr=LR,
        random=True,
        ratio_weighted=False,
        hard_mining=False,
        bsize=100,
        nval=0)


@RegisterExp('ours')
def ours_config():
    return Config(
        reweight=True,
        num_steps=NUM_STEPS,
        lr=LR,
        random=False,
        ratio_weighted=False,
        hard_mining=False,
        bsize=100,
        nval=FLAGS.nval)


def get_imbalance_dataset(mnist,
                          pos_ratio=0.9,
                          ntrain=5000,
                          nval=10,
                          ntest=500,
                          seed=0,
                          class_0=4,
                          class_1=9):
    rnd = np.random.RandomState(seed)

    # In training, we have 10% 4 and 90% 9.
    # In testing, we have 50% 4 and 50% 9.
    ratio = 1 - pos_ratio
    ratio_test = 0.5

    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels
    x_train_0 = x_train[y_train == class_0]
    x_test_0 = x_test[y_test == class_0]

    # First shuffle, negative.
    idx = np.arange(x_train_0.shape[0])
    rnd.shuffle(idx)
    x_train_0 = x_train_0[idx]

    nval_small_neg = int(np.floor(nval * ratio_test))
    ntrain_small_neg = int(np.floor(ntrain * ratio)) - nval_small_neg

    x_val_0 = x_train_0[:nval_small_neg]    # 450 4 in validation.
    x_train_0 = x_train_0[nval_small_neg:nval_small_neg + ntrain_small_neg]    # 500 4 in training.

    if FLAGS.verbose:
        print('Number of train negative classes', ntrain_small_neg)
        print('Number of val negative classes', nval_small_neg)

    idx = np.arange(x_test_0.shape[0])
    rnd.shuffle(idx)
    x_test_0 = x_test_0[:int(np.floor(ntest * ratio_test))]    # 450 4 in testing.

    x_train_1 = x_train[y_train == class_1]
    x_test_1 = x_test[y_test == class_1]

    # First shuffle, positive.
    idx = np.arange(x_train_1.shape[0])
    rnd.shuffle(idx)
    x_train_1 = x_train_1[idx]

    nvalsmall_pos = int(np.floor(nval * (1 - ratio_test)))
    ntrainsmall_pos = int(np.floor(ntrain * (1 - ratio))) - nvalsmall_pos

    x_val_1 = x_train_1[:nvalsmall_pos]    # 50 9 in validation.
    x_train_1 = x_train_1[nvalsmall_pos:nvalsmall_pos + ntrainsmall_pos]    # 4500 9 in training.

    idx = np.arange(x_test_1.shape[0])
    rnd.shuffle(idx)
    x_test_1 = x_test_1[idx]
    x_test_1 = x_test_1[:int(np.floor(ntest * (1 - ratio_test)))]    # 500 9 in testing.

    if FLAGS.verbose:
        print('Number of train positive classes', ntrainsmall_pos)
        print('Number of val positive classes', nvalsmall_pos)

    y_train_subset = np.concatenate([np.zeros([x_train_0.shape[0]]), np.ones([x_train_1.shape[0]])])
    y_val_subset = np.concatenate([np.zeros([x_val_0.shape[0]]), np.ones([x_val_1.shape[0]])])
    y_test_subset = np.concatenate([np.zeros([x_test_0.shape[0]]), np.ones([x_test_1.shape[0]])])

    y_train_pos_subset = np.ones([x_train_1.shape[0]])
    y_train_neg_subset = np.zeros([x_train_0.shape[0]])

    x_train_subset = np.concatenate([x_train_0, x_train_1], axis=0).reshape([-1, 28, 28, 1])
    x_val_subset = np.concatenate([x_val_0, x_val_1], axis=0).reshape([-1, 28, 28, 1])
    x_test_subset = np.concatenate([x_test_0, x_test_1], axis=0).reshape([-1, 28, 28, 1])

    x_train_pos_subset = x_train_1.reshape([-1, 28, 28, 1])
    x_train_neg_subset = x_train_0.reshape([-1, 28, 28, 1])

    # Final shuffle.
    idx = np.arange(x_train_subset.shape[0])
    rnd.shuffle(idx)
    x_train_subset = x_train_subset[idx]
    y_train_subset = y_train_subset[idx]

    idx = np.arange(x_val_subset.shape[0])
    rnd.shuffle(idx)
    x_val_subset = x_val_subset[idx]
    y_val_subset = y_val_subset[idx]

    idx = np.arange(x_test_subset.shape[0])
    rnd.shuffle(idx)
    x_test_subset = x_test_subset[idx]
    y_test_subset = y_test_subset[idx]

    train_set = DataSet(x_train_subset * 255.0, y_train_subset)
    train_pos_set = DataSet(x_train_pos_subset * 255.0, y_train_pos_subset)
    train_neg_set = DataSet(x_train_neg_subset * 255.0, y_train_neg_subset)
    val_set = DataSet(x_val_subset * 255.0, y_val_subset)
    test_set = DataSet(x_test_subset * 255.0, y_test_subset)

    return train_set, val_set, test_set, train_pos_set, train_neg_set


def get_exp_logger(sess, log_folder):
    """Gets a TensorBoard logger."""
    with tf.name_scope('Summary'):
        writer = tf.summary.FileWriter(os.path.join(log_folder), sess.graph)

    class ExperimentLogger():
        def log(self, niter, name, value):
            summary = tf.Summary()
            summary.value.add(tag=name, simple_value=value)
            writer.add_summary(summary, niter)

        def flush(self):
            """Flushes results to disk."""
            writer.flush()

    return ExperimentLogger()


def evaluate(sess, x_, y_, acc_, x, y, x_test, y_test):
    # Calculate final results.
    train_acc = sess.run(acc_, feed_dict={x_: x, y_: y})

    test_acc = sess.run(acc_, feed_dict={x_: x_test, y_: y_test})

    return train_acc, test_acc


def get_metric(pred, y):
    total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, pred)))
    R_squared = tf.reduce_mean(tf.subtract(1.0, tf.div(unexplained_error, total_error)))
    mse = tf.reduce_mean(tf.square(pred - y))
    return mse


def run(train_data, test_data, seed, task_name, target='label'):
    train_data, test_data, co_col, ca_col = data_preprocessing(train_data, test_data,
                                                               ca_co_sel_flag=False, onehot_flag=True)
    _, test_data = anomaly_dectection(train_data, test_data)
    # train_data, test_data = anomaly_dectection(train_data, test_data)# Outlier detection
    x, y, x_val, y_val, test_set, test_set_label = \
        get_dataset_(train_data, test_data, clean_ratio=clean_ratio,
                     test_retio=test_ratio, seed=seed, val_ratio=val_ratio)  # label confusion according to requirements
    x.reset_index(inplace=True)
    x.drop(columns=['基线-患者基本信息-ID_sparse'], inplace=True)
    y.reset_index(inplace=True)
    y_val = y.loc[y['sup_label'] == 0].sample(n=clean_data_num, random_state=seed)

    x_val = x.loc[y_val.index]
    x.drop(index=x_val.index, inplace=True)
    y.drop(index=x_val.index, inplace=True)

    ntrain = FLAGS.ntrain
    nval = FLAGS.nval
    ntest = FLAGS.ntest
    folder = os.path.join('ckpt_mnist_imbalance_cnn_p{:d}'.format(int(FLAGS.pos_ratio * 100.0)),
                          task_name + '_{:d}'.format(seed))
    if not os.path.exists(folder):
        os.makedirs(folder)

    with tf.Graph().as_default(), tf.Session() as sess:
        bsize = batchsize

        x_ = tf.placeholder(tf.float32, [None, x.shape[1]], name='x')
        y_ = tf.placeholder(tf.float32, [None], name='y')
        x_val_ = tf.placeholder(tf.float32, [None, x.shape[1]], name='x_val')
        y_val_ = tf.placeholder(tf.float32, [None], name='y_val')
        ex_wts_ = tf.placeholder(tf.float32, [None, 1], name='ex_wts')
        ex_wts_b = tf.placeholder(tf.float32, [None, 1], name='ex_wts_b')
        lr_ = tf.placeholder(tf.float32, [], name='lr')

        # Build training model.
        with tf.name_scope('Train'):
            _, loss_c, logits_c = get_model(
                x_, y_, is_training=True, dtype=tf.float32, w_dict=None, ex_wts=ex_wts_, reuse=None)
            train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss_c)
        # metric_ = get_metric(logits_c, y_)

        # Build evaluation model.
        with tf.name_scope('Val'):
            _, loss_eval, logits_eval = get_model(
                x_,
                y_,
                is_training=False,
                dtype=tf.float32,
                w_dict=None,
                ex_wts=ex_wts_,
                reuse=True)
            metric_ = get_metric(logits_eval, y_)

        # Build reweighting model.
        if reweight:
            if random:
                ex_weights_ = reweight_random(bsize)
            else:
                ex_weights_, var_list_, grads_, grads_w_ = reweight_autodiff(
                    x_,
                    y_,
                    x_val_,
                    y_val_,
                    ex_wts_,
                    ex_wts_b,
                    bsize,
                    clean_data_num,
                    eps=0.1,
                    gate_gradients=1)
        else:
            if hard_mining:
                ex_weights_ = reweight_hard_mining(x_, y_, positive=True)
            else:
                if ratio_weighted:
                    # Weighted by the ratio of each class.
                    ex_weights_ = pos_ratio * (1 - y_) + (1 - pos_ratio) * (y_)
                else:
                    # Weighted by uniform.
                    ex_weights_ = tf.ones([bsize], dtype=tf.float32) / float(bsize)

        if FLAGS.tensorboard:
            exp_logger = get_exp_logger(sess, folder)
        else:
            exp_logger = None
        num_steps = 10
        acc_sum = 0.0
        acc_test_sum = 0.0
        loss_sum = 0.0
        count = 0
        sess.run(tf.global_variables_initializer())
        history = pd.DataFrame([])
        history_loss = []
        history_loss_acc = []
        history_metric_r2 = []
        history_metric_mse = []
        history_metric_mae = []

        for i in range(2000):
            kf = KFold(n_splits=2, shuffle=False, random_state=2020)
            # for k, (train_index, val_index) in enumerate(kf.split(x)):
            #     x_batch, y_batch = x.iloc[train_index], y[target].iloc[train_index]
            x_batch, y_batch = x, y[target]
            ex_weights, var_list, grads, grads_w = sess.run(
                [ex_weights_, var_list_, grads_, grads_w_], feed_dict={x_: x_batch,
                                                                        y_: y_batch,
                                                                        x_val_: x_val,
                                                                        y_val_: y_val[target],
                                                                       ex_wts_: np.ones((batchsize, 1)),
                                                                       ex_wts_b: np.ones([clean_data_num, 1])})
            # ww = var_list[0]
            # bb = var_list[1]
            # print(x_batch.shape)
            # print(ww.shape)
            # xx = np.matmul(np.array(x_batch), ww)
            # xxx = xx + bb
            # xxxx = xxx - np.array(y_batch).reshape(-1, 1)
            # ss = (xxxx ** 2) / 2
            # sss = np.mean(ss)
            # ww_xx = xxxx.reshape(1, -1).dot(np.array(x_batch))
            # re_xx = np.mean(np.abs(xxxx))
            pred_tra, loss, acc, _ = sess.run(
                [logits_c, loss_c, metric_, train_op],
                feed_dict={
                    x_: x_batch,
                    y_: y_batch,
                    x_val_: x_val,
                    y_val_: y_val[target],
                    ex_wts_: ex_weights,
                    lr_: lr
                })
            print(np.unique(ex_weights))
            pred = sess.run(logits_eval, feed_dict={x_: test_set, y_: test_set_label[target], ex_wts_: ex_weights})
            r2 = r2_score(pred, test_set_label[target])
            mse = mean_squared_error(pred, test_set_label[target])
            mae = mean_absolute_error(pred, test_set_label[target])
            history_loss.append(loss)
            history_loss_acc.append(acc)
            history_metric_r2.append(r2)
            history_metric_mse.append(mse)
            history_metric_mae.append(mae)
        # Final evaluation.

        history['loss'] = history_loss
        history['acc'] = history_loss_acc
        history['r2'] = history_metric_r2
        history['mse'] = history_metric_mse
        history['mae'] = history_metric_mae

        pred_tra = sess.run(logits_eval, feed_dict={x_: x, y_: y[target], ex_wts_: ex_weights})
        train_r2 = r2_score(pred_tra, y[target])
        train_r2_ad = None
        train_mse = mean_squared_error(pred_tra, y[target])
        train_mae = mean_absolute_error(pred_tra, y[target])
        train_mape = None
        val_r2, val_r2_ad, val_mse, val_mae, val_mape, = None, None, None, None, None
        test_r2, test_r2_ad, test_mse, test_mae, test_mape = r2, None, mse, mae, None
        dict_ = dict(zip(['train_r2', 'train_r2_ad', 'train_mse', 'train_mae', 'train_mape',
                          'val_r2', 'val_r2_ad', 'val_mse', 'val_mae', 'val_mape',
                          'test_r2', 'test_r2_ad', 'test_mse', 'test_mae', 'test_mape'],
                         [train_r2, train_r2_ad, train_mse, train_mae, train_mape,
                          val_r2, val_r2_ad, val_mse, val_mae, val_mape,
                          test_r2, test_r2_ad, test_mse, test_mae, test_mape,
                          ]))
        metric_df = pd.DataFrame.from_dict([dict_])
    return metric_df, pd.DataFrame([]), pd.DataFrame([])


def main():
    metric_df_all = pd.DataFrame([])
    test_prediction_all = pd.DataFrame([]) # for prediction of test data
    history_df_all = pd.DataFrame([]) # for keras model
    for i, trial in enumerate(tqdm(six.moves.xrange(FLAGS.nrun))):
        print('rnum : {}'.format(i))
        seed = (trial * 2718) % 2020 # a different random seed for each run
        train_data, test_data = load_data_(datasets_name, task_name)
        metric_df, test_prediction, history_df = run(train_data, test_data, seed, task_name)
        metric_df_all = pd.concat([metric_df_all, metric_df], axis=0)
        test_prediction_all = pd.concat([test_prediction_all, test_prediction], axis=1)
        history_df_all = pd.concat([history_df_all, history_df], axis=1)

    for col in metric_df_all.columns:
        print('{} {:.4f} ({:.4f}) max: {:.4f} median {:.4f} min: {:.4f}'.format(col, metric_df_all[col].mean(),
                                                                                metric_df_all[col].std(),
                                                                                metric_df_all[col].max(),
                                                                                metric_df_all[col].median(),
                                                                                metric_df_all[col].min()))
    metric_df_all.to_csv('./metric_{}_{}nrun_{}Fold.csv'.format(task_name, nrun, n_splits), index=False)
    history_df_all.to_csv('./history_{}_{}nrun_{}Fold.csv'.format(task_name, nrun, n_splits), index=False)
    # test_prediction_all.columns = ['ab_time', 'ab_time_enh']
    test_prediction_all.to_csv('./prediction{}_{}nrun_{}Fold.csv'.format(task_name, nrun, n_splits))
    plt.show()
    pass


np.random.seed(2020)
datasets_name = 'LiverAblation'
task_name = 'ablation_time_learn_weight' # ablation_time_enh / ablation_time_vanilla / relapse_risk
nrun = 10 # num of repeated experiments
clean_ratio = 1 # 1 for No label confusion
test_ratio = 0 # test data ratio for label confusion
val_ratio = 0 # val data ratio for label confusion
n_splits = 1 # n_splits > 1 for Kfold cross validation / n_splits==1 for training all data
epoch = 5000 # Kfold cross validation: a large number / training all data: mean epoch
batchsize = 348
lr = 1e-4
clean_data_num = 10
reweight = True
num_steps = NUM_STEPS
random = False
ratio_weighted = False
hard_mining = False

if __name__ == '__main__':
    main()
