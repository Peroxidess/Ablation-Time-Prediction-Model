import tensorflow as tf
import numpy as np
import pandas as pd
from keras import backend as K
from keras import regularizers, activations
from keras.layers import Dense, Input, Add, Concatenate, Dropout, \
    BatchNormalization, Activation, Multiply, Embedding, Layer, GlobalAveragePooling1D
from keras.models import Model
import copy


class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape", WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (x.shape.as_list()[1] ** 0.5)

        QK = K.softmax(QK)

        print("QK.shape", QK.shape)

        V = K.batch_dot(QK, WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


class FM(Layer):
    def __init__(self, output_dim, latent=32,  activation='relu', **kwargs):
        self.latent = latent
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.b = self.add_weight(name='W0',
                                  shape=(self.output_dim,),
                                  trainable=True,
                                 initializer='zeros')
        self.w = self.add_weight(name='W',
                                 shape=(input_shape[1], self.output_dim),
                                 trainable=True,
                                 initializer='random_uniform')
        self.v= self.add_weight(name='V',
                                 shape=(input_shape[1], self.latent),
                                 trainable=True,
                                initializer='random_uniform')
        super(FM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        x_square = K.square(x)

        xv = K.square(K.dot(x, self.v))
        xw = K.dot(x, self.w)

        p = 0.5*K.sum(xv-K.dot(x_square, K.square(self.v)), 1)

        rp = K.repeat_elements(K.reshape(p, (-1, 1)), self.output_dim, axis=-1)

        f = xw + rp + self.b

        output = K.reshape(f, (-1, self.output_dim))

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape)==2
        return input_shape[0],self.output_dim


def Att(att_dim, inputs, name):
    V = inputs
    QK = Dense(att_dim//4, bias=None, activation='relu')(inputs)
    QK = Dense(att_dim, bias=None, activation='relu')(QK)
    QK = Activation("softmax", name=name)(QK)
    MV = Multiply()([V, QK])
    return(MV)


def regression_(train_x):
    input_dim = train_x.shape[1]
    l1_regul = 0
    l2_regul = 0
    input = Input(shape=(input_dim,))
    # input_ = BatchNormalization()(input, training=False)
    # input_fm = FM(input_dim)(input_)
    # input_emb = Embedding(input_dim + 1, input_dim//2)(input)
    # att = Self_Attention(input_dim//2)(input_emb)
    # att = GlobalAveragePooling1D()(att)
    atts1 = Att(input_dim, input, "attention_vec10")
    # atts11 = Att(input_dim, input_, "attention_vec11")
    # mlp_layer = Add()([atts1, atts11])
    # mlp_layer = Att(input_dim, mlp_layer, "attention_vec20")
    mlp_layer = atts1
    for units_ in [64, 16]:
        mlp_layer = Dense(units_, activation='relu',
                          kernel_regularizer=regularizers.l1_l2(l1=l1_regul, l2=l2_regul))(mlp_layer)
        # mlp_layer = Dropout(0.5)(mlp_layer)
        # mlp_layer = BatchNormalization()(mlp_layer, training=False)
    # atts2 = Att(32, mlp_layer, "attention_vec2")
    mlp_layer_output = Dense(1)(mlp_layer)
    regression = Model(input=input, output=mlp_layer_output)
    return regression


def classifer_(train_x):
    input_dim = train_x.shape[1]
    input_dim_emb = (input_dim + 1)
    input_ = Input(shape=(input_dim,))
    input_c = Input(shape=(1,))

    l1_regul = 0
    l2_regul = 0
    # encoder layers
    inputs = Concatenate()([input_, input_c])

    atts1 = Att(input_dim_emb, inputs, "attention_vec10")
    # atts2 = Att(input_dim + 1, inputs, "attention_vec11")
    # input_fm = FM(input_dim + 1)(atts1)
    encoded_layer = atts1
    # encoded_layer = Concatenate()([atts1, atts2])
    for units_ in [64]:
        encoded_layer = Dense(units_, activation='relu',
                              kernel_regularizer=regularizers.l1_l2(l1=l1_regul, l2=l1_regul))(encoded_layer)
        encoded_layer = Dropout(0.5)(encoded_layer)
        encoded_layer = BatchNormalization()(encoded_layer, training=False)

    encoder_output = Concatenate()([encoded_layer, input_c])

    # decoder layers
    decoded_layer = encoded_layer
    for units_ in [16, 128, train_x.shape[1]]:
        decoded_layer = Dense(units_, activation='relu',
                              kernel_regularizer=regularizers.l1_l2(l1=l1_regul, l2=l1_regul))(decoded_layer)
        # decoded_layer = Dropout(0.2)(decoded_layer)
        decoded_layer = BatchNormalization()(decoded_layer, training=False)

    # classifer layers
    classifer_layer = Dense(8, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1_regul, l2=l2_regul))(
        encoded_layer)
    classifer_layer = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=l1_regul, l2=l2_regul))(
        classifer_layer)

    # encoder = Model(input=[input_, input_c], output=encoded_layer)
    classifer = Model(input=[input_, input_c], output=classifer_layer)
    # autoencoder = Model(input=[input_, input_c], output=decoded_layer)
    att_weight = Model(input=[input_, input_c], output=atts1)
    # classifer.add_loss(recon_loss(y_true=input_, y_pred=decoded_layer))
    return classifer, att_weight


def eval_loss_and_grads(x, fetch_loss_and_grads):
    outs = fetch_loss_and_grads(x)
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def gradient_ascent(x, fetch_loss_and_grads, iter, step, max_loss=None, min_loss=None):
    """get gradient

    :param x:        [dataframe list] inputs and label
    :param fetch_loss_and_grads:     [ ]     K.function
    :param iter_:     [int]     Number of iterations for label modification
    :param step_:     [float]     Learning rate for label modification

    :return label_target:      [nparray]   Corrected label
    """
    for i in range(iter):
        loss_value, grad_values = eval_loss_and_grads(x, fetch_loss_and_grads)
        # if max_loss is not None and loss_value > max_loss:
        #     break
        x[1] = x[1] - step * np.squeeze(grad_values).reshape(-1, 1)
    return x


def label_correction(model, model_input, label, iter_=1, step_=1e-3):
    """correct label

    :param model:          [keras model] Relapse risk prediction model
    :param model_input:        [dataframe] Inputs
    :param label:     [series]     Labels that need to be corrected
    :param iter_:     [int]     Number of iterations for label modification
    :param step_:     [float]     Learning rate for label modification

    :return label_target:      [dataframe]   Corrected label
    """
    loss = K.variable(0.)
    coeff = 1
    activation = model.get_layer(index=-1).output
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss = loss + coeff * K.sum(K.square(activation[:, :])) / scaling
    dream = model.input
    grads = K.gradients(loss, dream[1])
    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
    outputs = [loss, grads]
    fetch_loss_and_grads = K.function([dream[0], dream[1]], outputs, K.set_learning_phase(0))
    label_target = pd.DataFrame(copy.deepcopy(label))
    label_target = gradient_ascent([model_input, label_target], fetch_loss_and_grads, iter=iter_, step=step_)[1]
    return label_target


def get_model(inputs,
              labels,
              is_training=True,
              dtype=tf.float32,
              w_dict=None,
              ex_wts=None,
              reuse=None):
    """
    :param inputs:            [Tensor]    Inputs.
    :param labels:            [Tensor]    Labels.
    :param is_training:       [bool]      Whether in training mode, default True.
    :param dtype:             [dtype]     Data type, default tf.float32.
    :param w_dict:            [dict]      Dictionary of weights, default None.
    :param ex_wts:            [Tensor]    Example weights placeholder, default None.
    :param reuse:             [bool]      Whether to reuse variables, default None.
    """

    if w_dict is None:
        w_dict = {}

    def _get_var(name, shape, dtype, initializer):
        key = tf.get_variable_scope().name + '/' + name
        if key in w_dict:
            return w_dict[key]
        else:
            var = tf.get_variable(name, shape, dtype, initializer=initializer)
            w_dict[key] = var
            return var

    with tf.variable_scope('Model', reuse=reuse):
        shape_list = np.append(np.array([-1]), np.squeeze(inputs.shape[1:].as_list()))
        # shape_list_fir = np.append(np.squeeze(inputs.shape[1:].as_list()), np.array([16]))
        # shape_list_sec = np.array([16, 8])
        # shape_list_thr = np.array([8, 1])
        inputs_ = tf.cast(tf.reshape(inputs, shape_list), dtype)
        labels = tf.cast(tf.reshape(labels, [-1, 1]), dtype)

        # w_init = tf.truncated_normal_initializer(stddev=0.1)
        # w1 = _get_var('w1', shape_list_fir, dtype, initializer=w_init)
        # w2 = _get_var('w2', shape_list_sec, dtype, initializer=w_init)
        # w3 = _get_var('w3', shape_list_thr, dtype, initializer=w_init)
        # w4 = _get_var('w4', [1, 1], dtype, initializer=w_init)
        #
        # b_init = tf.constant_initializer(0.0)
        # b1 = _get_var('b1', 1, dtype, initializer=b_init)
        # b2 = _get_var('b2', 1, dtype, initializer=b_init)
        # b3 = _get_var('b3', 1, dtype, initializer=b_init)
        # b4 = _get_var('b4', 1, dtype, initializer=b_init)
        #
        # act = tf.nn.relu
        #
        # l0 = tf.identity(inputs_, name='l0')
        # z1 = tf.add(tf.matmul(l0, w1), b1, name='z1')
        # l1 = act(z1, name='l1')
        # z2 = tf.add(tf.matmul(l1, w2), b2, name='z2')
        # l2 = act(z2, name='l2')
        # z3 = tf.add(tf.matmul(l2, w3), b3, name='z3')
        # l3 = act(z3, name='l3')
        # z4 = tf.add(tf.matmul(l3, w4), b4, name='z4')
        # logits = tf.squeeze(l3)
        # out = tf.sigmoid(logits)
        dense1 = tf.layers.dense(inputs=inputs_, units=64, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=16, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense2, units=1, activation=tf.nn.sigmoid)

        if ex_wts is None:
            # Average loss.
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        else:
            # Weighted loss.
            loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels) * ex_wts)
    return w_dict, loss, logits


def reweight_random(bsize, eps=0.0):
    """Reweight examples using random numbers.

    :param bsize:             [int]       Batch size.
    :param eps:               [float]     Minimum example weights, default 0.0.
    """
    ex_weight = tf.random_normal([bsize], mean=0.0, stddev=1.0)
    ex_weight_plus = tf.maximum(ex_weight, eps)
    ex_weight_sum = tf.reduce_sum(ex_weight_plus)
    ex_weight_sum += tf.to_float(tf.equal(ex_weight_sum, 0.0))
    ex_weight_norm = ex_weight_plus / ex_weight_sum
    return ex_weight_norm


def reweight_autodiff(inp_a,
                      label_a,
                      inp_b,
                      label_b,
                      bsize_a,
                      bsize_b,
                      eps=0.0,
                      gate_gradients=1):
    """Reweight examples using automatic differentiation.

    :param inp_a:             [Tensor]    Inputs for the noisy pass.
    :param label_a:           [Tensor]    Labels for the noisy pass.
    :param inp_b:             [Tensor]    Inputs for the clean pass.
    :param label_b:           [Tensor]    Labels for the clean pass.
    :param bsize_a:           [int]       Batch size for the noisy pass.
    :param bsize_b:           [int]       Batch size for the clean pass.
    :param eps:               [float]     Minimum example weights, default 0.0.
    :param gate_gradients:    [int]       Tensorflow gate gradients, reduce concurrency.
    """
    ex_wts_a = tf.zeros([bsize_a], dtype=tf.float32)
    ex_wts_b = tf.ones([bsize_b], dtype=tf.float32) / float(bsize_b)
    w_dict, loss_a, logits_a = get_model(
        inp_a, label_a, ex_wts=ex_wts_a, is_training=True, reuse=True)
    var_names = w_dict.keys()
    var_list = [w_dict[kk] for kk in var_names]
    grads = tf.gradients(loss_a, var_list, gate_gradients=gate_gradients)

    var_list_new = [vv - gg for gg, vv in zip(grads, var_list)]
    w_dict_new = dict(zip(var_names, var_list_new))
    _, loss_b, logits_b = get_model(
        inp_b, label_b, ex_wts=ex_wts_b, is_training=True, reuse=True, w_dict=w_dict_new)
    grads_ex_wts = tf.gradients(loss_b, [ex_wts_a], gate_gradients=gate_gradients)[0]
    ex_weight = -grads_ex_wts
    ex_weight_plus = tf.maximum(ex_weight, eps)
    ex_weight_sum = tf.reduce_sum(ex_weight_plus)
    ex_weight_sum += tf.to_float(tf.equal(ex_weight_sum, 0.0))
    ex_weight_norm = ex_weight_plus / ex_weight_sum
    return ex_weight_norm


def reweight_hard_mining(inp, label, positive=False):
    """Reweight examples using hard mining.

    :param inp:          [Tensor]   [N, ...] Inputs.
    :param label:        [Tensor]   [N] Labels
    :param positive:     [bool]     Whether perform hard positive mining or hard negative mining.

    :return              [Tensor]   Examples weights of the same shape as the first dim of inp.
    """
    _, loss, logits = get_model(inp, label, ex_wts=None, is_training=True, reuse=True)

    # Mine for positive
    if positive:
        loss_mask = loss * label
    else:
        loss_mask = loss * (1 - label)

    if positive:
        k = tf.cast(tf.reduce_sum(1 - label), tf.int32)
    else:
        k = tf.cast(tf.reduce_sum(label), tf.int32)
    k = tf.maximum(k, 1)
    loss_sorted, loss_sort_idx = tf.nn.top_k(loss_mask, k)

    if positive:
        mask = 1 - label
    else:
        mask = label

    updates = tf.ones([tf.shape(loss_sort_idx)[0]], dtype=label.dtype)
    mask_add = tf.scatter_nd(tf.expand_dims(loss_sort_idx, axis=1), updates, [tf.shape(inp)[0]])
    mask = tf.maximum(mask, mask_add)
    mask_sum = tf.reduce_sum(mask)
    mask_sum += tf.cast(tf.equal(mask_sum, 0.0), tf.float32)
    mask = mask / mask_sum
    return mask


def get_lenet_model(inputs,
              labels,
              is_training=True,
              dtype=tf.float32,
              w_dict=None,
              ex_wts=None,
              reuse=None):
    """Builds a simple LeNet.

    :param inputs:            [Tensor]    Inputs.
    :param labels:            [Tensor]    Labels.
    :param is_training:       [bool]      Whether in training mode, default True.
    :param dtype:             [dtype]     Data type, default tf.float32.
    :param w_dict:            [dict]      Dictionary of weights, default None.
    :param ex_wts:            [Tensor]    Example weights placeholder, default None.
    :param reuse:             [bool]      Whether to reuse variables, default None.
    """

    if w_dict is None:
        w_dict = {}

    def _get_var(name, shape, dtype, initializer):
        key = tf.get_variable_scope().name + '/' + name
        if key in w_dict:
            return w_dict[key]
        else:
            var = tf.get_variable(name, shape, dtype, initializer=initializer)
            w_dict[key] = var
            return var

    with tf.variable_scope('Model', reuse=reuse):
        inputs_ = tf.cast(tf.reshape(inputs, [-1, 28, 28, 1]), dtype)
        labels = tf.cast(labels, dtype)

        w_init = tf.truncated_normal_initializer(stddev=0.1)
        w1 = _get_var('w1', [5, 5, 1, 16], dtype, initializer=w_init)  # [14, 14, 16]
        w2 = _get_var('w2', [5, 5, 16, 32], dtype, initializer=w_init)  # [7, 7, 32]
        w3 = _get_var('w3', [5, 5, 32, 64], dtype, initializer=w_init)  # [4, 4, 64]
        w4 = _get_var('w4', [1024, 100], dtype, initializer=w_init)
        w5 = _get_var('w5', [100, 1], dtype, initializer=w_init)

        b_init = tf.constant_initializer(0.0)
        b1 = _get_var('b1', [16], dtype, initializer=b_init)
        b2 = _get_var('b2', [32], dtype, initializer=b_init)
        b3 = _get_var('b3', [64], dtype, initializer=b_init)
        b4 = _get_var('b4', [100], dtype, initializer=b_init)
        b5 = _get_var('b5', [1], dtype, initializer=b_init)

        act = tf.nn.relu

        # Conv-1
        l0 = tf.identity(inputs_, name='l0')
        z1 = tf.add(tf.nn.conv2d(inputs_, w1, [1, 1, 1, 1], 'SAME'), b1, name='z1')
        l1 = act(tf.nn.max_pool(z1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l1')

        # Conv-2
        z2 = tf.add(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'), b2, name='z2')
        l2 = act(tf.nn.max_pool(z2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l2')

        # Conv-3
        z3 = tf.add(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'), b3, name='z3')
        l3 = act(tf.nn.max_pool(z3, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l3')

        # FC-4
        z4 = tf.add(tf.matmul(tf.reshape(l3, [-1, 1024]), w4), b4, name='z4')
        l4 = act(z4, name='l4')

        # FC-5
        z5 = tf.add(tf.matmul(l4, w5), b5, name='z5')

        logits = tf.squeeze(z5)
        out = tf.sigmoid(logits)
        if ex_wts is None:
            # Average loss.
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        else:
            # Weighted loss.
            loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels) * ex_wts)
    return w_dict, loss, logits