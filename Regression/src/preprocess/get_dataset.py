from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
from preprocess import plot_tabel


def get_dataset_(nor, train_data, test_data, clean_ratio, test_retio, seed, target='label', val_ratio=0):
    if test_retio == 0 or test_data is not None:
        train_set = train_data
        test_set = test_data
    else:
        train_set, test_set = train_test_split(train_data, test_size=test_retio, random_state=seed)
    if clean_ratio < 1:
        train_set_, train_set_clean = train_test_split(train_set, test_size=clean_ratio, random_state=seed)
        label_distrib = np.random.normal(loc=train_set_[target].describe().loc['mean'],
                                         scale=train_set_[target].describe().loc['std'], size=train_set_[target].shape)
        alpha = 1
        beta = 1
        train_label_ = train_set_[target] + \
                       alpha * np.random.normal(loc=0., scale=1., size=train_set_[target].shape) + beta * label_distrib
        train_set_[target] = train_label_
        train_set_['sup_label'] = 1
        train_set_clean['sup_label'] = 0
        test_set['sup_label'] = 0
    else:
        train_set_ = None
        train_set_clean = train_set

    train_set_mix = pd.concat([train_set_, train_set_clean], axis=0)
    # mix_ratio = train_set[train_set[target] != train_set_mix[target]].index
    # print('real mix ratio is {}'.format(mix_ratio))
    if val_ratio > 0:
        train_set_mix, val_set = train_test_split(train_set_mix, test_size=val_ratio, random_state=seed)
        val_set_label = val_set[[target, 'sup_label']]
        val_set.drop(columns=[target, 'sup_label'], inplace=True)
    else:
        val_set = None
        val_set_label = None
    train_set_mix_label = train_set_mix[[target, 'sup_label']]
    test_set_label = test_set[[target, 'sup_label']]
    # plot_tabel.metric_hist(test_set, nor)
    train_set_mix.drop(columns=[target, 'sup_label'], inplace=True)
    test_set.drop(columns=[target, 'sup_label'], inplace=True)
    return train_set_mix, train_set_mix_label, val_set, val_set_label, test_set, test_set_label


def data_preprocessing(train_data, test_data=None, ca_feat_th=8, ca_co_sel_flag=True, onehot_flag=False, target='label'):
    if test_data is not None:
        train_data['tab'] = 1
        test_data['tab'] = 0
    data_raw = pd.concat([train_data, test_data], axis=0)
    print('\ndata_raw', data_raw.shape)
    data = data_raw.dropna(axis=1, how='all')
    xx = data.isnull().sum()
    data = data.fillna(0)
    if ca_co_sel_flag:
        ca_col = []
        co_col = []
        data_columns_label = data.filter(regex=r'label').columns
        data_columns = data.columns.drop(data_columns_label)
        # data_columns = data.columns.drop(['sup_label'])
        for col in data_columns:
            data_col = data[col]
            col_feat_num = len(set(data_col))
            if col_feat_num > ca_feat_th:
                col_ = col + '_dense'
                co_col.append(col_)
                data.rename(columns={col: col_}, inplace=True)
            elif ca_feat_th >= col_feat_num > 1:
                col_ = col + '_sparse'
                ca_col.append(col_)
                data.rename(columns={col: col_}, inplace=True)
    else:
        ca_col = data.filter(regex=r'sparse').columns
        co_col = data.filter(regex=r'dense').columns
    data[ca_col] = pd.concat([data[ca_col].apply(lambda ser: pd.factorize(ser)[0])])
    data[ca_col] = data[ca_col].apply(LabelEncoder().fit_transform)
    if onehot_flag:
        data = pd.get_dummies(data, columns=ca_col)
    co_col = co_col.append(data.columns[data.columns == target]) # 回归目标也需要归一化避免在sup_label分类预测中的模型崩溃
    mms = MinMaxScaler(feature_range=(0.1, 1.1))
    std = StandardScaler()
    xx = data.filter(regex=r'label').describe()
    xx_col = xx.index
    xx_min = xx.loc['min', :]
    xx_max = xx.loc['max', :]
    xx_std = xx.loc['std', :]
    data[co_col] = pd.DataFrame(std.fit_transform(data[co_col]), columns=co_col, index=data.index)
    # data[co_col] = pd.DataFrame(mms.fit_transform(data[co_col]), columns=co_col, index=data.index)
    # data = pd.DataFrame(mms.fit_transform(data), columns=data.columns, index=data.index)
    if test_data is not None:
        train_data = data[data['tab'] == 1].drop(columns=['tab'])
        test_data = data[data['tab'] == 0].drop(columns=['tab'])
    else:
        train_data = data
    ca_col = data.filter(regex=r'sparse').columns
    co_col = data.filter(regex=r'dense').columns
    return train_data, test_data, co_col, ca_col, std


def anomaly_dectection(train_data=None, test_data=None, target='label'):
    clean_data = []
    for data in [train_data, test_data]:
        if not data.empty:
            std_ = data[target].std()
            mean_ = data[target].mean()
            data = data[data[target] < mean_ + 3 * std_]
            data = data[data[target] > mean_ - 3 * std_]
        clean_data.append(data)
    return clean_data[0], clean_data[1]

