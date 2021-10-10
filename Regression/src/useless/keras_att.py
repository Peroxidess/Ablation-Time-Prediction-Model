import pandas as pd
import numpy as np
from tqdm import tqdm
import six
import tensorflow as tf
from keras import losses
from keras import backend as K
from keras import optimizers
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Multiply, Activation, Layer, \
    GlobalAveragePooling1D, Reshape, RepeatVector, Flatten, Lambda, Add, Embedding
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    confusion_matrix, precision_score, recall_score, f1_score, r2_score
import matplotlib.pyplot as plt
from preprocess.load_data import load_data_
from preprocess.get_dataset import get_dataset_, foo, anomaly_dectection


class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, 1, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)

    def call(self, x):
        x = K.expand_dims(x, axis=2)
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape", WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (x.shape.as_list()[-1] ** 0.5)

        QK = K.softmax(QK)

        print("QK.shape", QK.shape)

        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    activations = []
    input = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([input] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def r2(y_true, y_pred):
    return 1 - K.sum(K.square(y_pred - y_true))/K.sum(K.square(y_true - K.mean(y_true)))


def r_square(y_true, y_pred):
    SSR = K.mean(K.square(y_pred-K.mean(y_true)), axis=-1)
    SST = K.mean(K.square(y_true-K.mean(y_true)), axis=-1)
    return SSR/SST


def Att(att_dim, inputs, name):
    V = inputs
    QK = Dense(att_dim, bias=None)(inputs)
    QK = Dense(att_dim, bias=None)(QK)
    QK = Activation("softmax", name=name)(QK)
    MV = Multiply()([V, QK])
    return(MV)


def bulid_model(train_set_mix, train_set_mix_label, ca_col, co_col):
    input_dim = train_set_mix.shape[-1]
    inputs = Input(shape=(input_dim,))

    atts1 = Att(input_dim, inputs, "attention_vec")
    x = Dense(64, activation='relu')(atts1)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    # atts2 = Att(4, atts2, "attention_vec1")
    output = Dense(1)(x)
    model = Model(input=inputs, output=output)
    return model


def Expand_Dim_Layer(tensor):
    def expand_dim(tensor):
        return K.expand_dims(tensor, axis=1)
    return Lambda(expand_dim)(tensor)


def bulid_model_atts(train_set_mix, train_set_mix_label, ca_col, co_col):
    input_dim = train_set_mix.shape[-1]
    inputs_ = Input(shape=(input_dim,))
    # inputs_emb = Embedding(10000, input_dim)(inputs_)
    atts1 = Self_Attention(input_dim)(inputs_)
    atts1 = GlobalAveragePooling1D()(atts1)
    x = Dense(64, activation='relu')(atts1)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs_, outputs=outputs)
    model.summary()
    return model


def run(train_data, test_data, seed, reg_flag=False, label_enh_flag=False, reg_enh_flag=False, target='label'):
    train_data, test_data, co_col, ca_col = foo(train_data, test_data, ca_co_sel_flag=False, onehot_flag=True)
    train_set_mix, train_set_mix_label, val_set, val_set_label, test_set, test_set_label = \
        get_dataset_(train_data, test_data, clean_ratio=clean_ratio,
                     test_retio=test_ratio, seed=seed, val_ratio=val_ratio)

    train_curr_label = train_set_mix_label[target]
    test_curr_label = test_set_label[target]
    model = bulid_model_atts(train_set_mix, train_set_mix_label, ca_col, co_col)
    rms = optimizers.RMSprop(lr=1e-4)
    model.compile(optimizer=rms, loss='mean_squared_error', metrics=['mse', 'mae', r2, r_square])
    model.fit(train_set_mix, train_curr_label, epochs=epoch, batch_size=batchsize, validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss', patience=200, min_delta=0.01)])

    train_target_pred = model.predict(train_set_mix)
    test_target_pred = model.predict(test_set)
    num_data = train_set_mix.shape[0]
    num_feat = train_set_mix.shape[1]
    train_r2 = r2_score(train_set_mix_label[target], train_target_pred)
    train_r2_ad = 1 - ((1 - train_r2) * (num_data - 1)) / abs(num_data - num_feat - 1)
    test_r2 = r2_score(test_set_label[target], test_target_pred)
    test_r2_ad = 1 - ((1 - test_r2) * (num_data - 1)) / abs(num_data - num_feat - 1)
    train_mse = mean_squared_error(train_set_mix_label[target], train_target_pred)
    train_mae = mean_absolute_error(train_set_mix_label[target], train_target_pred)
    test_mse = mean_squared_error(test_set_label[target], test_target_pred)
    test_mae = mean_absolute_error(test_set_label[target], test_target_pred)
    test_mape = mean_absolute_percentage_error(test_set_label[target], test_target_pred.reshape(-1, ))
    err_enh = test_set_label[target] - np.squeeze(test_target_pred)

    # attention_vector = get_activations(model, train_set_mix[:1],
    #                                    print_shape_only=True,
    #                                    layer_name='attention_vec')[0].flatten()
    # pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',
    #                                                                title='Attention Mechanism as a '
    #                                                                      'function of input dimensions.')
    # plt.show()
    return test_r2, test_r2_ad, test_mse


def run_many(train_data, test_data):
    metric_list_all = []
    for trial in tqdm(six.moves.xrange(nrun)):
        # train_metric, test_metric, train_metric_enh, test_metric_enh = \
        #     run(train_data, test_data, (trial * 2020) % 1000, reg_flag=True, label_enh_flag=True, reg_enh_flag=True)
        metric_list = run(train_data, test_data, (trial * 2020) % 1000,
                          reg_flag=True, label_enh_flag=True, reg_enh_flag=True)
        metric_list_all.append(metric_list)

    metric_df = pd.DataFrame(np.array(metric_list_all))
    metric_df.columns = ['train_metric', 'train_metric_enh', 'test_metric', 'test_metric_enh']
    for col in metric_df.columns:
        print('{} metric {:.3f} ({:.3f}) max: {:.3f}'.format(col, metric_df[col].mean(),
                                                             metric_df[col].std(),
                                                             metric_df[col].max()))
    pass


def main():
    train_data, test_data = load_data_(datasets_name)
    run_many(train_data, test_data)
    pass


np.random.seed(2020)
datasets_name = 'LiverAblation'
nrun = 5
clean_ratio = 1
test_ratio = 0.2
val_ratio = 0
epoch = 3000
batchsize = 16
iter_ = 10
step_ = 0.001

if __name__ == '__main__':
    main()
