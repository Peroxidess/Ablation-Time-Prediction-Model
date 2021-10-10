import pandas as pd
import numpy as np
from tqdm import tqdm
import six
import tensorflow as tf
from keras import losses
from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    confusion_matrix, precision_score, recall_score, f1_score, r2_score
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, RidgeCV, LassoCV, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
# from deepctr.models import DeepFM, xDeepFM, DCN, WDL
# from deepctr.feature_column import SparseFeat, get_feature_names, DenseFeat
from preprocess.load_data import load_data_
from preprocess.get_dataset import get_dataset_, data_preprocessing, anomaly_dectection
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def ctr_model(linear_feature_columns, dnn_feature_columns):
    adam = tf.keras.optimizers.Adam(lr=0.0001)
    model = WDL(linear_feature_columns, dnn_feature_columns, task='regression')
    # model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
    model.compile(adam, "huber_loss", metrics=['mae'],)
    return model


def baseline_model(train_set_mix, train_set_mix_label, ca_col, co_col, seed):
    clf = lgb.LGBMRegressor(max_depth=3,
                            bagging_fraction=0.7,
                            feature_fraction=0.7,
                            reg_alpha=0.5,
                            reg_lambda=0.5,
                            min_child_samples=10,
                            n_estimators=200,
                            learning_rate=1e-1,
                            random_state=seed,
                            )
    # clf = lgb.LGBMRegressor(max_depth=4,
    #                         bagging_fraction=0.8,
    #                         feature_fraction=0.8,
    #                         reg_alpha=0.8,
    #                         reg_lambda=0.8,
    #                         min_child_samples=10,
    #                         n_estimators=500,
    #                         learning_rate=1e-1,
    #                         )

    # clf = lgb.LGBMRegressor()
    # clf = LassoCV()
    # clf = RidgeCV()
    return clf


def run(train_data, test_data, seed, target='label'):
    np.random.seed(seed)
    train_data, test_data, co_col, ca_col = data_preprocessing(train_data, test_data,
                                                ca_co_sel_flag=False, onehot_flag=False)
    # train_data, _ = anomaly_dectection(train_data, test_data=pd.DataFrame())
    # _, test_data = anomaly_dectection(train_data=pd.DataFrame(), test_data=test_data)
    # train_data, test_data = anomaly_dectection(train_data=train_data, test_data=test_data)
    train_set_mix, train_set_mix_label, val_set, val_set_label, test_set, test_set_label = \
        get_dataset_(train_data, test_data, clean_ratio=clean_ratio,
                     test_retio=test_ratio, val_ratio=val_ratio, seed=seed)

    # fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=pd.concat([train_set_mix, test_set], axis=0)[feat].nunique(), embedding_dim=4)
    #                        for i, feat in enumerate(ca_col)] + [DenseFeat(feat, 1,)
    #                       for feat in co_col]
    #
    # dnn_feature_columns = fixlen_feature_columns
    # linear_feature_columns = fixlen_feature_columns
    # feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    # train_set_mix = {name: train_set_mix[name].values for name in feature_names}
    # test_set = {name: test_set[name].values for name in feature_names}
    # model = ctr_model(linear_feature_columns, dnn_feature_columns,)
    # history = model.fit(train_set_mix, train_set_mix_label[target].values,
    #                     batch_size=512, epochs=180, verbose=1, validation_split=0.2, )
    # train_set_mix = train_set_mix.loc[train_set_mix_label['sup_label'] == 0]
    # train_set_mix_label = train_set_mix_label.loc[train_set_mix_label['sup_label'] == 0]
    model = baseline_model(train_set_mix, train_set_mix_label, ca_col, co_col, seed)
    model.fit(train_set_mix, train_set_mix_label[target])

    # feat_df = pd.DataFrame({'column': train_set_mix.columns, 'importance': model.feature_importances_.round(5)})
    # feat_df_sort = feat_df.sort_values(by='importance', ascending=False)
    # feat_df_sort_ = feat_df_sort.set_index(['column'])
    # feat_df_sort_[:30].plot.barh(figsize=(15, 15), fontsize=12)
    # plt.title("n61_lgb_特征重要性")
    # plt.show()
    train_target_pred = model.predict(train_set_mix)
    test_target_pred = model.predict(test_set)
    train_R2 = r2_score(train_set_mix_label[target], train_target_pred)
    num_data = train_set_mix.shape[0]
    num_feat = train_set_mix.shape[1]
    train_R2_ad = 1 - ((1 - train_R2) * (num_data - 1)) / abs(num_data - num_feat - 1)
    test_R2 = r2_score(test_set_label[target], test_target_pred)
    num_data = test_set.shape[0]
    num_feat = test_set.shape[1]
    test_R2_ad = 1 - ((1 - test_R2) * (num_data - 1)) / abs(num_data - num_feat - 1)
    train_mse = mean_squared_error(train_set_mix_label[target], train_target_pred)
    train_mae = mean_absolute_error(train_set_mix_label[target], train_target_pred)
    test_mse = mean_squared_error(test_set_label[target], test_target_pred)
    test_mae = mean_absolute_error(test_set_label[target], test_target_pred)
    test_mape = mean_absolute_percentage_error(test_set_label[target], test_target_pred.reshape(-1, ))
    err = test_set_label[target] - np.squeeze(test_target_pred)
    return [train_R2, test_R2, train_R2_ad, test_R2_ad, train_mse, test_mse, train_mae, test_mae, test_mape]


def run_many(train_data, test_data):
    metric_list_all = []
    for trial in tqdm(six.moves.xrange(nrun)):
        metric_list = run(train_data, test_data, (trial * 2718) % 2020)
        metric_list_all.append(metric_list)

    metric_df = pd.DataFrame(np.array(metric_list_all))
    metric_df.columns = ['train_R2', 'test_R2',
                         'train_R2_ad', 'test_R2_ad',
                         'train_mse', 'test_mse',
                         'train_mae', 'test_mae',
                         'test_mape',]
    for col in metric_df.columns:
        print('{} {:.4f} ({:.4f}) max: {:.4f} min: {:.4f}'.format(col, metric_df[col].mean(),
                                                                  metric_df[col].std(),
                                                                  metric_df[col].max(),
                                                                  metric_df[col].min()))
    pass


def main():
    train_data, test_data = load_data_(datasets_name)
    run_many(train_data, test_data)
    pass


datasets_name = 'LiverAblation'
nrun = 10
clean_ratio = 1
test_ratio = 0.2
val_ratio = 0.2
epoch = 200
batchsize = 1
iter_ = 1
step_ = 0.1

if __name__ == '__main__':
    main()
