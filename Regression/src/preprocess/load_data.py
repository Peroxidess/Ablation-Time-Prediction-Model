#coding=gb18030
import numpy as np
import pandas as pd


def load_data_(datasets, task_name='', seed=2020):
    if datasets == 'winequality_white':
        data_path = '../DataSet/wine/{}.csv'.format(datasets)
        data = pd.read_csv(data_path)
        data.rename(columns={'quality': 'label'}, inplace=True)
        data.dropna(axis=0, subset=['label'], inplace=True)
        train_data = data.fillna(0)
        test_data = None
    elif datasets == 'PPH':
        data_path = '../DataSet/PPH/{}.csv'.format(datasets)
        data_head = pd.read_csv('../DataSet/PPH/PPH_head.csv', encoding='gb18030')
        data = pd.read_csv(data_path, encoding='gb18030', index_col='index')

        col = []
        for col_ in data.columns:
            col.append(col_ + np.squeeze(data_head[col_].values))
        data.columns = np.array(col)
        # data.to_csv('../DataSet/PPH/data_feat_name_add.csv', index=False, encoding='gb18030')

        data['sup_label'] = 0
        label_col = data.filter(regex=r'n61').columns.values[0]
        data.rename(columns={label_col: 'label'}, inplace=True)
        data.dropna(axis=0, subset=['label'], inplace=True)
        data['hours'] = data.filter(regex=r'field12').values - data.filter(regex=r'field9').values
        data['hours'] = data['hours'].apply(lambda x: 24 + x if x < 0 else x)
        data['minutes'] = data.filter(regex=r'field13').values - data.filter(regex=r'field10').values
        data['minutes'] = data['minutes'].apply(lambda x: 60 + x if x < 0 else x)
        data['minutes'] += data['hours'] * 60
        drop_columns = data.filter(
            regex=r'n421|field11|其他|field28|其他.1|n262|hours|n61|n51|n4417|n4318|field9|field10|field12|field13').columns
        train_data = data.drop(columns=drop_columns)
        # data.fillna(0, inplace=True)
        test_data = None
    elif datasets == 'LiverAblation':
        data_path = '../DataSet/LiverAblation/{}.csv'.format(datasets)
        data = pd.read_csv(data_path, encoding='gb18030', index_col='基线-患者基本信息-ID_sparse')
        # data_path = '../DataSet/LiverAblation/{}_trans.csv'.format(datasets)
        # data = pd.read_csv(data_path, encoding='gb18030', index_col='baseline_info_ID_sparse')

        data.rename(columns={'time_dense': 'label'}, inplace=True)
        data.rename(columns={'relapse_sparse': 'sup_label'}, inplace=True)
        drop_columns_ = data.filter(regex=r'随|ID|cluster|followupInfomation').columns
        data.drop(columns=drop_columns_, inplace=True)

        data_1 = data.loc[data['sup_label'] == 1]
        data_0 = data.loc[data['sup_label'] == 0].sample(n=data_1.shape[0] * 1, random_state=seed)
        data_undersmapling = pd.concat([data_1, data_0]).sample(frac=1, random_state=seed)
        test_data = data.drop(index=data_undersmapling.index)
        if 'non' in task_name:
            train_data = data_0
        else:
            train_data = data_undersmapling
    else:
        train_data = None
        test_data = None
    return train_data, test_data
