import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    confusion_matrix, precision_score, recall_score, f1_score, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler


def evaluate_classification(model, train_sets, train_label, val_sets, val_label, test_sets, test_label):
    relapse_risk_test = model.predict(test_sets)
    relapse_risk_tra = model.predict(train_sets)
    con_mat = confusion_matrix(test_label, relapse_risk_test.round())
    train_acc = accuracy_score(train_label, relapse_risk_tra.round())
    test_acc = accuracy_score(test_label, relapse_risk_test.round())
    train_f1 = f1_score(train_label, relapse_risk_tra.round())
    test_f1 = f1_score(test_label, relapse_risk_test.round())
    val_acc = None
    val_f1=None
    if val_label is not None:
        relapse_risk_val = model.predict(val_sets)
        val_acc = accuracy_score(val_label, relapse_risk_val.round())
        val_f1 = f1_score(val_label, relapse_risk_val.round())
    dict_ = dict(zip(['train_acc', 'test_acc', 'val_acc', 'val_f1', 'train_f1', 'test_f1'],
                     [train_acc, test_acc, val_acc, val_f1, train_f1, test_f1]))
    return pd.DataFrame([dict_])


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def wmape(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true)) * 100


def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


def evaluate_regression(model, train_sets, train_label, val_x, val_label, test_sets, test_label):
    test_target_pred = model.predict(test_sets)
    train_target_pred = model.predict(train_sets)
    num_data_tra = train_sets.shape[0]
    num_feat_tra = train_sets.shape[1]
    num_data_test = train_sets.shape[0]
    num_feat_test = train_sets.shape[1]
    train_r2 = r2_score(train_label, train_target_pred)
    train_r2_ad = 1 - ((1 - train_r2) * (num_data_tra - 1)) / abs(num_data_tra - num_feat_tra - 1)
    test_r2 = r2_score(test_label, test_target_pred)
    test_r2_ad = 1 - ((1 - test_r2) * (num_data_test - 1)) / abs(num_data_test - num_feat_test - 1)
    train_mse = mean_squared_error(train_label, train_target_pred)
    train_mae = mean_absolute_error(train_label, train_target_pred)
    test_mse = mean_squared_error(test_label, test_target_pred)
    test_mae = mean_absolute_error(test_label, test_target_pred)

    mms = MinMaxScaler(feature_range=(0.1, 1))
    train_label_mms = mms.fit_transform(np.array(train_label).reshape(-1, 1))
    test_label_mms = mms.fit_transform(np.array(test_label).reshape(-1, 1))
    train_target_pred_mns = mms.fit_transform(train_target_pred.reshape(-1, 1))
    test_target_pred_mns = mms.fit_transform(test_target_pred.reshape(-1, 1))
    train_mape = wmape(train_label_mms, train_target_pred_mns.reshape(-1, ))
    test_mape = wmape(test_label_mms, test_target_pred_mns.reshape(-1, ))
    err = test_label - np.squeeze(test_target_pred)
    if not val_x.empty:
        val_target_pred = model.predict(val_x)
        num_data_val = val_x.shape[0]
        num_feat_val = val_x.shape[1]
        val_r2 = r2_score(val_label, val_target_pred)
        val_r2_ad = 1 - ((1 - val_r2) * (num_data_val - 1)) / abs(num_data_val - num_feat_val - 1)
        val_mse = mean_squared_error(val_label, val_target_pred)
        val_mae = mean_absolute_error(val_label, val_target_pred)
        val_label_mms = mms.fit_transform(np.array(val_label).reshape(-1, 1))
        val_target_pred_mns = mms.fit_transform(val_target_pred.reshape(-1, 1))
        val_mape = smape(val_label_mms, val_target_pred_mns.reshape(-1, ))
    else:
        val_r2, val_r2_ad, val_mse, val_mae, val_mape = None, None, None, None, None
    dict_ = dict(zip(['train_r2', 'train_r2_ad', 'train_mse', 'train_mae', 'train_mape',
                      'val_r2', 'val_r2_ad', 'val_mse', 'val_mae', 'val_mape',
                      'test_r2', 'test_r2_ad', 'test_mse', 'test_mae', 'test_mape'],
                     [train_r2, train_r2_ad, train_mse, train_mae, train_mape,
                      val_r2, val_r2_ad, val_mse, val_mae, val_mape,
                      test_r2, test_r2_ad, test_mse, test_mae, test_mape,
                      ]))
    return pd.DataFrame.from_dict([dict_])
