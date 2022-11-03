import random, os, sys
import numpy as np
import tensorflow as tf
# from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.python.keras.layers import *
from  tensorflow.python.keras.models import *
try:
    from tqdm import tqdm
    from dataloader import TokenList, pad_to_longest
# for transformer
except:
    pass



# It's safe to use a 1-d mask for self-attention
class ScaledDotProductAttention(Layer):
    def __init__(self, attn_dropout=0.1):
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):  # mask_k or mask_qk
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)([q, k])  # shape=(batch, q, k)
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention(Layer):
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, dropout, mode=1):
        self.d_model=d_model
        self.mode = mode
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention()
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, s[2] // n_head])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], s[2] // n_head])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []
            attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head)
                attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        return outputs, attn

    def get_config(self):
        config = {"n_head": self.n_head,"d_model":self.d_model,"dropout":self.dropout,"mode":self.mode}
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionwiseFeedForward(Layer):
    def __init__(self, d_hid, d_inner_hid, mode='dense',dropout=0.1):
        self.d_hid=d_hid
        self.d_inner_hid=d_inner_hid
        if mode=='conv':
            self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
            self.w_2 = Conv1D(d_hid, 1)
        else:
            self.w_1 = Dense(d_inner_hid, activation='relu')
            self.w_2 = Dense(d_hid)
        self.layer_norm = LayerNormalization(epsilon=1e-2)
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)



class EncoderLayer(Layer):
    def __init__(self, d_model, d_inner_hid, n_head, mode, residual_dropout=0.1, attention_dropout=0.1,**kwargs):
        self.d_model=d_model
        self.d_inner_hid=d_inner_hid
        self.n_head=n_head
        self.mode=mode
        self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=attention_dropout,mode=1)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, mode, dropout=residual_dropout)
        self.norm_layer = LayerNormalization(epsilon=1e-2)
        super(EncoderLayer, self).__init__(**kwargs)


    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)  # MultiHeadAttention
        output = self.norm_layer(Add()([enc_input, output]))  # Add & Norm
        output = self.pos_ffn_layer(output)   #Feed Forward + Add & Norm
        return output, slf_attn

    def get_config(self):
        config = {"d_model": self.d_model,"d_inner_hid":self.d_inner_hid,"n_head":self.n_head,"dropout":self.dropout}
        base_config = super(EncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))