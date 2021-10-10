import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
plt.rc('font', family='Times New Roman')
font_size = 16


def plot_metric_df(history_list, task_name, val_flag='test_'):
    if 'relapse_risk' in task_name:
        metric_list = ['loss', 'f1']
    else:
        metric_list = ['r2', 'mae', 'mse']
    fig = plt.figure(figsize=(20, 4))
    L = len(metric_list)
    row = math.floor(math.sqrt(L))
    col = L / row
    for i, metric in enumerate(metric_list):
        plt.subplot(row, col, i+1)
        show_metric(history_list, metric, val_flag)
        fig.subplots_adjust(top=0.8)
    legend_labels = ['ours',
                        # 'enh_nonrelapse',
                        'ATT+MLP',
                        # 'vanilla_nonrelapse',
                        'LGB',
                        # 'lightgbm_nonrelapse',
                        'Lasso',
                        # 'lasso_nonrelapse'
                        ]
    plt.legend(labels= legend_labels,
                ncol = len(legend_labels),
                # loc='best', 
                loc='upper center', 
                fontsize=14,
                bbox_to_anchor=(-1.2, 1, 1, 0.2),
                borderaxespad = 0.,
                )
        # plt.title('{} {}'.format(task_name, metric), fontsize=font_size)


def show_metric(history_list, metrics_name, val_flag=''):
    marker_list = ['*', 'd', 's', 'x', 'o']
    metrics_name_dict = {'r2':'R-square', 'mae':'mean absolute error', 'mse':'mean squared error'}

    for m, history in enumerate(history_list):
        history_metric = history.filter(regex=r'\b{}{}\b'.format(val_flag, metrics_name))[:3000]

        plt.plot(history_metric, linestyle=':', marker=marker_list[m], linewidth=2)

    plt.xticks(range(0, 11), fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel(metrics_name_dict[metrics_name], fontsize=font_size)
    plt.xlabel('Round', fontsize=font_size)


def plot_history_df(history_list, task_name, val_flag=''):
    if 'relapse_risk' in task_name:
        metric_list = ['loss', 'f1']
    else:
        metric_list = ['loss', 'r2']
    plt.figure(figsize=(12, 4))
    L = len(metric_list)
    row = math.floor(math.sqrt(L))
    col = L / row
    for i, metric in enumerate(metric_list):
        plt.subplot(row, col, i+1)
        show_history(history_list, metric, val_flag)
        plt.legend(labels=['attention', 'attention+mlp', 'attention+label corrected',
                           'attention+mlp+label corrected(ours)', 'mlp', 'mlp+label corrected'],
                   fontsize=14)
        # plt.title('{} {}'.format(metric, task_name), fontsize=font_size)


def show_history(history_list, metrics_name, val_flag=''):
    marker_list = ['^', 'd', 's', '*', 'x', 'o']
    for m, history in enumerate(history_list):
        history_metric = history.filter(regex=r'\b{}{}'.format(val_flag, metrics_name))[:3000]
        history_ = np.mean(history_metric, axis=1)
        len_ = history_.shape[0]
        plt.plot(history_, linewidth=2, marker=marker_list[m], markevery=200)
        plt.fill_between(range(len_), np.min(history_metric, axis=1), np.max(history_metric, axis=1), alpha=0.3)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel(val_flag + metrics_name, fontsize=font_size)
    plt.xlabel('Epoch', fontsize=font_size)


def plot_history(history_list, task_name, val_flag=False):
    if task_name == 'relapse_risk':
        metric_list = ['loss', 'f1']
    else:
        metric_list = ['loss', 'r2']
    plt.figure(figsize=(12, 4))
    L = len(metric_list)
    for i, metric in enumerate(metric_list):
        plt.subplot(squrt(), L, i+1)
        show_train_history(history_list, metric)
        if val_flag:
            show_train_history(history_list, 'val_{}'.format(metric))
        plt.legend(labels=[metric, 'val_{}'.format(metric)], loc='upper left')
        plt.title('{} {}'.format(task_name, metric))


def history_save(history_list, history_name):
    history_all = pd.DataFrame([])
    for history in history_list:
        history_ = pd.DataFrame.from_dict(history.history, orient='index')
        history_all = pd.concat([history_all, history_], axis=0)
    history_all.to_csv('./hitory_{}.csv'.format(history_name))


def show_train_history(history_list, metrics_name):
    metrics_list = None
    for history in history_list:
        history_metric = pd.DataFrame(np.array(history.history[metrics_name]).reshape(1, -1))
        if metrics_list is None:
            metrics_list = history_metric
        else:
            metrics_list = pd.concat([metrics_list, history_metric], axis=0)
    # metrics = np.median(metrics_list, axis=0)
    metrics = np.mean(metrics_list, axis=0)
    plt.plot(metrics)
    plt.ylabel(metrics_name)
    plt.xlabel('Epoch')
