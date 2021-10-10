import numpy as np
import pandas as pd
import six
from tqdm import tqdm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from preprocess.load_data import load_data_
from preprocess.get_dataset import get_dataset_, data_preprocessing, anomaly_dectection
from model.training_ import training_model, model_training, precision, recall, f1, r2
from model.history_ import plot_history_df


def run(train_data, test_data, seed, task_name, target='label'):
    train_data, test_data, co_col, ca_col, nor = data_preprocessing(train_data, test_data,
                                                                    ca_co_sel_flag=False, onehot_flag=True)
    _, test_data = anomaly_dectection(train_data, test_data)
    # train_data, test_data = anomaly_dectection(train_data, test_data)# Outlier detection
    train_set_mix, train_set_mix_label, val_set, val_set_label, test_set, test_set_label = \
        get_dataset_(nor,train_data, test_data, clean_ratio=clean_ratio,
                     test_retio=test_ratio, seed=seed, val_ratio=val_ratio,)# label confusion according to requirements
    metric_df = pd.DataFrame([])
    test_prediction = pd.DataFrame([])
    history_df = pd.DataFrame([])
    history_list = []
    epoch_len_list = []

    if n_splits > 1:
        kf = KFold(n_splits=n_splits, shuffle=False, random_state=seed)
        for k, (train_index, val_index) in enumerate(kf.split(train_set_mix)):
            print('KFlod in : {}'.format(k))
            model_, history_, metric_, test_pred_, epoch_len = training_model(train_set_mix, train_set_mix_label,
                                                                              task_name, train_index, val_index,
                                                                              test_set, test_set_label,
                                                                              epoch, batchsize, iter_, step_, target, seed)
            metric_df = pd.concat([metric_df, metric_], axis=0)
            history_df = pd.concat([history_df, history_], axis=1)
            history_list.append(history_)
            test_prediction = pd.concat([test_prediction, pd.DataFrame(test_pred_)], axis=1)
            epoch_len_list.append(epoch_len)
        plot_history_df(history_list, task_name)
        print('epoch_len_mean', np.mean(epoch_len_list)) # mean epoch in kflod cross validation

    else:
        model_, history_, metric_, test_pred_, epoch_len = training_model(train_set_mix, train_set_mix_label,
                                                                          task_name, None, None,
                                                                          test_set, test_set_label,
                                                                          epoch, batchsize, iter_, step_, target, seed)
        metric_df = pd.concat([metric_df, metric_], axis=0)
        test_prediction = pd.concat([test_prediction, pd.DataFrame(test_pred_)], axis=1)
        history_df = pd.concat([history_df, history_], axis=1)
        history_list.append(history_)
        plot_history_df(history_list, task_name, val_flag='val_')
    try:
        model_.save('{}_{}nrun_{}Fold.h5'.format(task_name, nrun, n_splits))
    except:
        print('Failed to save model')
    return metric_df, test_prediction, history_df


np.random.seed(2020)
datasets_name = 'LiverAblation'
task_name = 'ablation_time_load' # ablation_time_enh / ablation_time_vanilla / relapse_risk
nrun = 10 # num of repeated experiments
clean_ratio = 1 # 1 for No label confusion
test_ratio = 0 # test data ratio for label confusion
val_ratio = 0 # val data ratio for label confusion
n_splits = 1 # n_splits > 1 for Kfold cross validation / n_splits==1 for training all data
epoch = 5000 # Kfold cross validation: a large number / training all data: mean epoch
batchsize = 256
iter_ = 2 # Number of iterations for label modification
step_ = 0.0001 # learning rate for label modification


def main():
    metric_df_all = pd.DataFrame([])
    test_prediction_all = pd.DataFrame([]) # for prediction of test data
    history_df_all = pd.DataFrame([]) # for keras model
    for i, trial in enumerate(tqdm(six.moves.xrange(nrun))):
        print('rnum : {}'.format(i))
        seed = (trial * 2718) % 2020 # a different random seed for each run
        train_data, test_data = load_data_(datasets_name, task_name,seed)
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


if __name__ == '__main__':
    main()
    pass
