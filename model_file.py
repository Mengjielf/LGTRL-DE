import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from Transformer_Encoder import EncoderLayer
from Transformer_Encoder import MultiHeadAttention as multi_head
from keras.regularizers import *
import keras.backend as K
from keras import initializers, regularizers, constraints
from tensorflow.python.framework import dtypes, function
from tensorflow.python.ops import math_ops, array_ops, nn
import numpy as np
import math

#
def local_attention(inputs, single_attention_vector=False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    conv_a = Conv1D(filters=input_dim, kernel_size=3, name="Local", padding='causal')(inputs)
    a = Permute((2, 1))(conv_a)  # (batch_size, input_dim, time_steps)
    a = Dense(time_steps, activation='softmax')(a)  # (batch_size, input_dim, time_steps)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)  # (batch_size,  time_steps, input_dim,)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


class PositionEncoding(Layer):

    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        seq_length = inputs.shape[1]
        position_encodings = np.zeros((seq_length, self._model_dim))
        for pos in range(seq_length):
            for i in range(self._model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i - i % 2) / self._model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2])  # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2])  # 2i+1
        position_encodings = K.cast(position_encodings, 'float32')
        # print("position_encodings：", position_encodings.shape)
        return position_encodings

    def compute_output_shape(self, input_shape):
        return input_shape



def LGTRL_DE(time_steps, N, input_dims, emb_size, trans_emb_size, demo_emb_size, d_inner_hid, gru_units, output_dim):
    inputs = []
    x = Input(shape=(time_steps, input_dims[0]))
    inputs.append(x)

    emb_x = Dense(emb_size, activation='relu', name='emb_x')(x)
    emb_x = Dropout(rate=0.5)(emb_x)

    demo = Input(shape=(input_dims[1],))
    inputs.append(demo)
    demo = Dense(demo_emb_size, use_bias=False, activation='relu', name='demo_emb')(demo)

    # LSTM
    lstm = emb_x
    lstm = Concatenate(axis=1)([Reshape((1, -1))(demo), lstm])
    lstm = Bidirectional(CuDNNGRU(units=gru_units, return_sequences=True, name='gru'))(lstm)
    lstm_att = local_attention(lstm)

    # Transformer_Encoder
    position = PositionEncoding(trans_emb_size)(lstm_att)
    coding_x = lstm_att + position
    transformer_depth = N  #
    hidden_state = [None for i in range(transformer_depth)]
    trans = coding_x
    for i in range(transformer_depth):
        trans, hidden_state[i] = EncoderLayer(d_model=trans_emb_size, d_inner_hid=d_inner_hid, n_head=2,
                                              mode='dense', attention_dropout=0.5, residual_dropout=0.5,
                                              name="trans" + str(i))(trans)
    trans = Lambda(lambda x: x, name="trans")(trans)

    final_h = Concatenate()([lstm_att, trans])
    final_h = Bidirectional(CuDNNGRU(units=gru_units, return_sequences=False, name='gru2'))(final_h)
    output = Concatenate()([final_h, demo])
    output = Dropout(rate=0.5)(output)
    output = Dense(output_dim, activation="sigmoid")(output)

    return Model(inputs=inputs, outputs=output)