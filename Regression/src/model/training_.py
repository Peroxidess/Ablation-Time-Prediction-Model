import copy
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, RidgeCV, LassoCV, LinearRegression
from keras.models import load_model
from keras import backend as K
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping
from model.bulid_model import classifer_, regression_, label_correction
from model.evaluate import evaluate_classification, evaluate_regression


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def r2(y_true, y_pred):
    return 1 - K.sum(K.square(y_pred - y_true))/K.sum(K.square(y_true - K.mean(y_true)))


def f1(y_true, y_pred):
    return 2 * precision(y_true, y_pred) * \
           recall(y_true, y_pred) / (precision(y_true, y_pred) + recall(y_true, y_pred) + 1e-7)


# model compile and fit
def model_training(model, train_sets, train_label, val_data, val_label, lr, task, epoch, batch_size, patience=100):
    if task == 'classification':
        metrics = ['acc', f1, precision, recall]
        loss = 'binary_crossentropy'
        val_metric = 'val_f1'
    elif task == 'regression':
        metrics = ['mse', 'mae', r2]
        metrics = [r2]
        loss = 'mean_squared_error'
        val_metric = 'val_r2'
    model.compile(optimizer=RMSprop(lr=lr), loss=loss, metrics=metrics)
    model.summary()
    if val_label is None:
        history = model.fit(train_sets, train_label,
                            epochs=epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            callbacks=[EarlyStopping(monitor=val_metric, patience=patience, mode='max')],
                            # callbacks=[EarlyStopping(monitor='val_loss', patience=200, min_delta=0.01)],
                            verbose=2,
                            )
    else:
        history = model.fit(train_sets, train_label,
                            # validation_split=0.3,
                            validation_data=(val_data, val_label),
                            epochs=epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            callbacks=[EarlyStopping(monitor=val_metric, patience=patience, mode='max')],
                            # callbacks=[EarlyStopping(monitor='val_loss', patience=200, min_delta=0.01)],
                            verbose=2,
                            )
    return history, model


# select model
def training_model(train_set, train_set_label, task_name, train_index, val_index, test_set, test_set_label,
                   epoch, batchsize, iter_=None, step_=None, target='label', seed=2020, label_corr_epoch=2):
    if train_index is not None:
        train_x, val_x = train_set.iloc[train_index], train_set.iloc[val_index]
        train_y, val_y = train_set_label.iloc[train_index], train_set_label.iloc[val_index]
        val_label = val_y[target]
        val_suplabel = val_y['sup_label']
        val_x_time = val_x.drop(columns=val_x.filter(regex=r'术后|出院|Post').columns)
    else:
        train_x = train_set
        train_y = train_set_label
        val_x = test_set
        val_x_time = test_set.drop(columns=val_x.filter(regex=r'术后|出院|Post').columns)
        val_label = test_set_label[target]
        val_suplabel = test_set_label['sup_label']
    train_x_time = train_x.drop(columns=train_x.filter(regex=r'术后|出院|Post').columns)
    test_set_time = test_set.drop(columns=test_set.filter(regex=r'术后|出院|Post').columns)
    # train_x_time.to_csv('train_data.csv', encoding='gb18030')
    train_data_raw = pd.read_csv('train_data.csv', encoding='gb18030')
    xx = set(train_data_raw.columns) - set(train_x_time.columns)
    rr = set(train_x_time.columns) - set(train_data_raw.columns)
    if 'risk' in task_name:
        classifer, att_weight = classifer_(train_x)
        # epoch=130 for training whole data 107
        # lr=8e-5 batchsize=8 patience= 90
        history, model = model_training(classifer,
                                        [train_x, train_y[target]], train_y['sup_label'],
                                        [val_x, val_label], val_suplabel,
                                        8e-5, 'classification', 120, 16, 190)
        metric = evaluate_classification(model,
                                         [train_x, train_y[target]], train_y['sup_label'],
                                         [val_x, val_label], val_suplabel,
                                         [test_set, test_set_label[target]], test_set_label['sup_label'])
        test_pred = model.predict([test_set, test_set_label[target]])
        history_df = pd.DataFrame.from_dict(history.history, orient='columns')
        len_ = history_df.shape[0] # count the number of epoch

    elif 'vanilla' in task_name:
        regression = regression_(train_x_time)
        # epoch=2926 for training whole data 2709 for non-relapse data
        # lr=9e-6 batchsize=256 patience= 350
        history, model = model_training(regression, train_x_time, train_y[target], val_x_time, val_label,
                                        9e-6, 'regression', 15000, batchsize, 2500)  #240 2335
        metric = evaluate_regression(model, train_x_time, train_y[target],
                                     val_x_time, val_label,
                                     test_set_time, test_set_label[target],
                                     )
        test_pred = model.predict(test_set_time)
        history_df = pd.DataFrame.from_dict(history.history, orient='columns')
        len_ = len(history.history['loss'])  # count the number of epoch
    elif 'load' in task_name:
        model = load_model('ablation_time_enh_10nrun_1Fold.h5', custom_objects={'r2': r2})
        test_pred = model.predict(test_set_time)
        history_df = pd.DataFrame([])
        metric = evaluate_regression(model,
                                     train_x_time, train_y[target],
                                     val_x_time, val_label,
                                     test_set_time, test_set_label[target],
                                     )
        len_ = 0
    elif 'enh' in task_name:
        history_df = pd.DataFrame([])
        classifer, att_weight = classifer_(train_x)
        # lr=8e-5 batchsize=16 epoch= 120
        history, classifer = model_training(classifer,
                                        [train_set, train_set_label[target]], train_set_label['sup_label'],
                                        [pd.DataFrame([]), None], None,
                                        8e-5, 'classification', 120, 16, 130)
        label_target = copy.deepcopy(train_set_label[target])
        regression_enh = regression_(train_x_time)
        len_ = 0
        for i in range(label_corr_epoch):
            print('iter {}'.format(i))
            label_target = label_correction(classifer, train_set, label_target, iter_=iter_, step_=step_)
            # label_target = train_y[target]
            if train_index is not None:
                label_target_train = label_target.iloc[train_index]
                val_label = label_target.iloc[val_index]
            else:
                label_target_train = label_target

            # lr=9e-6 batchsize=256 epoch= 600
            history, model = model_training(regression_enh,
                                            train_x_time, label_target_train, val_x_time, val_label,
                                           7e-5, 'regression', 225, batchsize, 220,)
                                           # 1e-5, 'regression', 1750, batchsize, 2120, )
            metric = evaluate_regression(model,
                                         train_x_time, train_y[target],
                                         val_x_time, val_label,
                                         test_set_time, test_set_label[target],
                                         )
            test_pred = model.predict(test_set_time)
            if history_df.empty:
                history_df = pd.DataFrame.from_dict(history.history, orient='columns')
            else:
                history_df = pd.concat([history_df, pd.DataFrame.from_dict(history.history, orient='columns')], axis=0)
            len_ += history_df.shape[0] # count the number of epoch
        history_df.reset_index(drop=True, inplace=True)
        if train_index is not None:
            val_pred = model.predict(val_x_time)
            risk = classifer.predict([val_x, train_set_label[target].iloc[val_index]])
            risk_corr = classifer.predict([val_x, val_pred])
            risk_change = risk - risk_corr
            risk_change_max = risk_change.max()
            risk_change_mean = risk_change.mean()
            x = 1
    elif 'lr' in task_name:
        model = LassoCV(random_state=seed)
        # model = RidgeCV()
        model.fit(train_x_time, train_y[target])
        metric = evaluate_regression(model,
                                     train_x_time, train_y[target],
                                     val_x_time, val_label,
                                     test_set_time, test_set_label[target],
                                     )
        history_df = pd.DataFrame([])
        len_ = 0
        test_pred = model.predict(test_set_time)
    elif 'gbm' in task_name:
        model = lgb.LGBMRegressor(
            max_depth=3,
            bagging_fraction=0.5,
            feature_fraction=0.5,
            reg_alpha=1,
            reg_lambda=1,
            min_child_samples=10,
            n_estimators=200,
            learning_rate=1e-1,
            random_state=seed,
                                  )
        model.fit(train_x_time, train_y[target])
        metric = evaluate_regression(model,
                                     train_x_time, train_y[target],
                                     val_x_time, val_label,
                                     test_set_time, test_set_label[target],
                                     )
        history_df = pd.DataFrame([])
        len_ = 0
        test_pred = model.predict(test_set_time)
    return model, history_df, metric, test_pred, len_
