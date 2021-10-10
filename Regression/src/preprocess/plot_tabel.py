import copy
import pandas as pd
import matplotlib.pyplot as plt
from model.history_ import plot_history_df, plot_metric_df
import numpy as np
from scipy.stats import ttest_ind, levene
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def metric_hist(data, nor=None):
    root_ = '../report/result/'
    file_list = [
        'ablation_time_enh_1nrun_10Fold.csv',# ours

        # 'ablation_time_vanilla_att_only__1nrun_10Fold.csv',# att only
        # 'ablation_time_vanilla_natt_1nrun_10Fold.csv',#mlp only

        # 'ablation_time_enh_att_only__10nrun_1Fold.csv',#
        'ablation_time_enh_natt_1nrun_10Fold.csv',# mlp+lc

        'lr_10nrun_1Fold.csv',# baseline_lasso
        'lr_non_1nrun_10Fold.csv',# nonrelapse
        'gbm_1nrun_10Fold.csv',# gbm
        'gbm_non_1nrun_10Fold.csv',# nonrelapse

        'ablation_time_vanilla_1nrun_10Fold.csv',# ATT+MLP
        'ablation_time_vanilla_non_1nrun_10Fold.csv',# att+mlp+non relapse

        # 'ablation_time_learn_weight_10nrun_1Fold.csv',

        # 'ablation_time_enh_non_10nrun_1Fold.csv',  # 0.2297
        # 'ablation_time_vanilla_att_only_10nrun_1Fold.csv',#

        # 'ablation_time_enh_natt__10nrun_1Fold.csv',# 0.5686
        # 'ablation_time_enh_att_only__10nrun_1Fold.csv',# 0.5690
        # 'ablation_time_enh_natt__10nrun_1Fold.csv',#  0.5686
                         ]
    metric_file_list = ['metric_' + file for file in file_list]
    history_file_list = ['history_' + file for file in file_list]
    pred_file_list = ['prediction' + file for file in file_list]

    tt_pvalue_list = np.array([])
    lv_pvalue_list = np.array([])
    metric_file_base = metric_file_list[0]
    metric_df_base = pd.read_csv(root_ + metric_file_base)

    for metric_file in metric_file_list:
        metric_df = pd.read_csv(root_ + metric_file)
        mae_col = metric_df.filter(regex=r'mae').columns
        mse_col = metric_df.filter(regex=r'mse').columns
        # metric_df[mae_col] = metric_df.loc[:, mae_col] * 562.062540
        # metric_df[mse_col] = metric_df.loc[:, mse_col] * 562.062540**2
        print('\n', metric_file)
        for col in metric_df.columns:
            print('{} {:.4f} ({:.4f}) max: {:.4f} median {:.4f} min: {:.4f}'.format(col, metric_df[col].mean(),
                                                                                    metric_df[col].std(),
                                                                                    metric_df[col].max(),
                                                                                    metric_df[col].median(),
                                                                                    metric_df[col].min()))

        v1 = metric_df_base['test_mae']
        v2 = metric_df['test_mae']
        std_ = levene(v1, v2).pvalue
        lv_pvalue_list = np.append(lv_pvalue_list, std_)
        equal_var_ = False
        if std_ > 0.05:
            equal_var_ = True
        res = ttest_ind(v1, v2, equal_var=equal_var_).pvalue
        tt_pvalue_list = np.append(tt_pvalue_list, res)
    tt_pvalue_list = tt_pvalue_list.reshape(-1, 1)

    for pred_file in pred_file_list:
        pred_df = pd.read_csv(root_ + pred_file, index_col=0)
        data_inver_label_df = pd.DataFrame([])
        metric_df = pd.DataFrame([])
        for pred in pred_df:
            data_co = data.filter(regex=r'dense|^label')
            data_ = copy.deepcopy(data_co)
            data_.loc[:, 'label'] = np.array(pred_df[pred])
            data_inver_pred = pd.DataFrame(nor.inverse_transform(data_), columns=data_.columns)
            data_inver = pd.DataFrame(nor.inverse_transform(data_co), columns=data_co.columns)
            data_inver_pred_label = data_inver_pred['label']
            data_inver_label = data_inver['label']
            mae = mean_absolute_error(data_inver_label, data_inver_pred_label)
            mse = mean_squared_error(data_inver_label, data_inver_pred_label)
            mape_ = mape(data_inver_label, data_inver_pred_label)
            r2 = r2_score(data_inver_label, data_inver_pred_label)
            dict_ = dict(zip([
                              'test_r2', 'test_mse', 'test_mae', 'test_mape'],
                             [
                              r2, mse, mae, mape_,
                              ]))
            metric_ = pd.DataFrame.from_dict([dict_])
            metric_df = pd.concat([metric_df, metric_], axis=0)
            data_inver_label_df = pd.concat([data_inver_label_df, data_inver_label], axis=1)
            # data_inver.to_csv(root_ + 'inver' + pred_file)

    history_df_all_list = []
    for history_file in history_file_list:
        history_df_all = pd.read_csv(root_ + history_file)
        history_df_all_list.append(history_df_all)
    # plot_history_df(history_df_all_list, task_name='ablation_time', val_flag='')
    plot_history_df(history_df_all_list, task_name='of the experimental results of ablation time prediction ', val_flag='val_')
    plt.show()

    metric_df_all_list = []
    metric_file_list = ['metric_ablation_time_enh_10nrun_1Fold.csv',
                        # 'metric_ablation_time_enh_non_10nrun_1Fold.csv',
                        'metric_ablation_time_vanilla_10nrun_1Fold.csv',
                        # 'metric_ablation_time_vanilla_non_10nrun_1Fold.csv',
                        'metric_gbm_10nrun_1Fold.csv',
                        # 'metric_gbm_non_10nrun_1Fold.csv',
                        'metric_lr_10nrun_1Fold.csv',
                        # 'metric_lr_non_10nrun_1Fold.csv',
                        ]
    for history_file in metric_file_list:
        history_df_all = pd.read_csv(root_ + history_file)
        metric_df_all_list.append(history_df_all)
    # plot_history_df(history_df_all_list, task_name='ablation_time', val_flag='')
    plot_metric_df(metric_df_all_list, task_name='ablation_time', val_flag='test_')
    plt.show()
    pass