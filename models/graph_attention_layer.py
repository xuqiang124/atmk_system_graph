from __future__ import absolute_import

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU
import tensorflow as tf


class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.5,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)
    
    def get_config(self):
        config = super(GraphAttention, self).get_config()
        config.update({
            'F_': self.F_,
            'attn_heads': self.attn_heads,
            'attn_heads_reduction': self.attn_heads_reduction,
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'attn_kernel_initializer': initializers.serialize(self.attn_kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'attn_kernel_regularizer': regularizers.serialize(self.attn_kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'attn_kernel_constraint': constraints.serialize(self.attn_kernel_constraint),
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    # def call(self, inputs):
    #     X = inputs[0]  # Node features (N x F)
    #     A = inputs[1]  # Adjacency matrix (N x N)

    #     outputs = []
    #     for head in range(self.attn_heads):
    #         kernel = self.kernels[head]  # W in the paper (F x F')
    #         attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

    #         # Compute inputs to attention network
    #         features = K.dot(X, kernel)  # (N x F')

    #         # Compute feature combinations
    #         # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
    #         attn_for_self = K.dot(features, attention_kernel[0])    # (N x 1), [a_1]^T [Wh_i]
    #         attn_for_neighs = K.dot(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

    #         # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
    #         dense = attn_for_self + K.transpose(attn_for_neighs)  # (N x N) via broadcasting

    #         # Add nonlinearty
    #         dense = LeakyReLU(alpha=0.2)(dense)

    #         # Mask values before activation (Vaswani et al., 2017)
    #         mask = -10e9 * (1.0 - A)
    #         dense += mask

    #         # Apply softmax to get attention coefficients
    #         dense = K.softmax(dense)  # (N x N)

    #         # Apply dropout to features and attention coefficients
    #         dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
    #         dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')

    #         # Linear combination with neighbors' features 
    #         # 在矩阵乘法 K.dot(dropout_attn, dropout_feat) 中，第 i 行的新特征是所有节点特征的加权和，权重由 dropout_attn[i,:] 决定
    #         # 因此，在 GAT 中，邻接矩阵 A[i,j] 控制的是从节点 j 到节点 i 的信息流动。
    #         node_features = K.dot(dropout_attn, dropout_feat)  # (N x F')

    #         if self.use_bias:
    #             node_features = K.bias_add(node_features, self.biases[head])

    #         # Add output of attention head to final output
    #         outputs.append(node_features)

    #     # Aggregate the heads' output according to the reduction method
    #     if self.attn_heads_reduction == 'concat':
    #         output = K.concatenate(outputs)  # (N x KF')
    #     else:
    #         output = K.mean(K.stack(outputs), axis=0)  # N x F')

    #     output = self.activation(output)
    #     return output

    # def compute_output_shape(self, input_shape):
    #     output_shape = input_shape[0][0], self.output_dim
    #     return output_shape

    def call(self, inputs):
        X = inputs[0]  # (batch_size, N, F)
        A = inputs[1]  # (batch_size, N, N)
        
        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]
            attention_kernel = self.attn_kernels[head]
            
            # 使用tf.einsum处理批处理维度
            features = tf.einsum('bnf,fh->bnh', X, kernel)  # (batch_size, N, F')
            
            attn_for_self = tf.einsum('bnf,fa->bna', features, attention_kernel[0])  # (batch_size, N, 1)
            attn_for_neighs = tf.einsum('bnf,fa->bna', features, attention_kernel[1])  # (batch_size, N, 1)
            
            # 处理注意力权重
            dense = attn_for_self + tf.transpose(attn_for_neighs, perm=[0, 2, 1])  # (batch_size, N, N)
            
            # 添加非线性
            dense = tf.nn.leaky_relu(dense, alpha=0.2)
            
            # 掩码处理
            mask = -10e9 * (1.0 - A)
            dense += mask
            
            # 应用softmax
            dense = tf.nn.softmax(dense, axis=-1)  # (batch_size, N, N)
            
            # 应用dropout
            dropout_attn = tf.nn.dropout(dense, rate=self.dropout_rate)  # (batch_size, N, N)
            dropout_feat = tf.nn.dropout(features, rate=self.dropout_rate)  # (batch_size, N, F')
            
            # 线性组合
            node_features = tf.einsum('bnn,bnf->bnf', dropout_attn, dropout_feat)  # (batch_size, N, F')
            
            if self.use_bias:
                node_features = tf.nn.bias_add(node_features, self.biases[head])
            
            outputs.append(node_features)
        
        # 聚合多头注意力的输出
        if self.attn_heads_reduction == 'concat':
            output = tf.concat(outputs, axis=-1)  # (batch_size, N, K*F')
        else:
            output = tf.reduce_mean(tf.stack(outputs, axis=1), axis=1)  # (batch_size, N, F')
        
        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        # 更新输出形状计算
        return (input_shape[0][0], input_shape[0][1], self.output_dim)