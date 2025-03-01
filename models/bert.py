import keras
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LayerNormalization, Dropout
from keras.layers import Flatten, Concatenate, Permute, Lambda, Dot
import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from transformers import TFAutoModel


np.random.seed(3407)
tf.random.set_seed(3407)

class Classifier(object):
    """
    分类器
    """
    @classmethod
    def build(self, config, use_att=False, label_emb_matrix=None, metrics=None):

        maxlen = config.maxlen
        hidden_size = config.hidden_size
        num_classes_list = config.num_classes_list

        # 输入层
        input_ids = Input(shape=(maxlen,), dtype='int32', name='input_ids')
        attention_mask = Input(shape=(maxlen,), dtype='int32', name='attention_mask')

        # BERT 编码器层
        bert = TFAutoModel.from_pretrained("bert-base-chinese")
        bert_output = bert(input_ids=input_ids, attention_mask=attention_mask)[0]  # [0]表示最后一层隐藏状态 #shape (None, max_len. hidden_size)
        

       # 计算所有标签数目
        hierarchy_levels=len(num_classes_list)
        count = sum(num_classes_list[i] for i in range(hierarchy_levels))
        label_input = Input(shape=(count,), name='label_input') # 标签
       
        bert_output = bert_output[:, 0, :] # 使用 [CLS] token 的隐藏状态作为分类输入

        pred_probs = Dense(num_classes_list[-1], activation='sigmoid',
                           name='pred_probs')(bert_output)

        model = Model(inputs=[input_ids,attention_mask, label_input], outputs=pred_probs)
        optimizer = Adam(learning_rate=2e-5)
        # 每一批次评估一次
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer, metrics=metrics)  # 自定义评价函数

        model._get_distribution_strategy = lambda: None  # fix bug for 2.1 tensorboard
        print(model.summary())
        return model, bert_output, bert_output
