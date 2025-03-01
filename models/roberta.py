import keras
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LayerNormalization, Dropout
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from transformers import TFBertModel
# 导入 GraphAttention 层
from models.graph_attention_layer import GraphAttention


np.random.seed(3407)
tf.random.set_seed(3407)

class Classifier(object):
    """
    分类器
    """
    @classmethod
    def build(cls, config, use_att=False, label_emb_matrix=None, metrics=None):
        """
        构建分类器模型
        
        Args:
            config: 配置对象，包含模型参数
            use_att: 是否使用注意力机制
            label_emb_matrix: 标签嵌入矩阵
            metrics: 评估指标
            
        Returns:
            model: 构建的模型
            roberta_output: RoBERTa输出
            label_emb_last: 最后一层标签嵌入
        """
        # 从配置中提取参数
        maxlen = config.maxlen
        hidden_size = config.hidden_size
        num_classes_list = config.num_classes_list
        labels_relation = config.labels_relation
        hierarchy_levels = len(num_classes_list)
        
        def _attention(input_x, label_emb_matrix, name=""):
            """
            注意力层
            
            Args:
                input_x: [batch_size, max_length, hidden_size] 输入特征
                label_emb_matrix: [batch_size, num_classes, hidden_size] 标签嵌入矩阵
                name: 作用域名称
                
            Returns:
                attention_weight_probs: [batch_size, num_classes, max_length] 注意力权重
                attention_out: [batch_size, hidden_size] 注意力输出
                weighted_text_representation: [batch_size, num_classes, hidden_size] 加权文本表示
            """
            # 计算注意力权重
            attention_weight = tf.matmul(
                label_emb_matrix, 
                tf.transpose(input_x, perm=[0, 2, 1]), 
                name=f"{name}_weight"
            )
            attention_weight_probs = tf.sigmoid(attention_weight)
            
            # 应用注意力权重 weighted_text_representation: (batch_size, num_classes, hidden_size)
            weighted_text_representation = tf.matmul(attention_weight_probs, input_x)
            
            # 聚合加权表示 attention_out: (batch_size, hidden_size)
            attention_out = tf.reduce_mean(
                weighted_text_representation, 
                axis=1, 
                name=f"{name}_context"
            )
            
            # 规范化和dropout
            attention_out = LayerNormalization()(attention_out)
            attention_out = Dropout(0.1)(attention_out)
            
            return attention_weight_probs, attention_out, weighted_text_representation

        def _local_layer(input_x, input_att_weight, num_classes, name=""):
            """
            局部层
            
            Args:
                input_x: [batch_size, hidden_size] 输入特征
                input_att_weight: [batch_size, num_classes, sequence_length] 注意力权重
                num_classes: 类别数量
                name: 作用域名称
                
            Returns:
                visual: [batch_size, sequence_length] 预测传递
            """
            # 计算类别得分
            scores = Dense(
                num_classes, 
                activation='sigmoid', 
                name=f"{name}_scores"
            )(input_x)
            
            # 应用得分并计算softmax
            visual = tf.multiply(input_att_weight, tf.expand_dims(scores, -1))# [batch_size, num_classes, sequence_length]
            visual = tf.nn.softmax(visual)# [batch_size, num_classes, sequence_length]
            visual = tf.reduce_mean(visual, axis=1, name="visual")# [batch_size, sequence_length]
            
            return visual
        
        # 构建模型输入
        input_ids = Input(shape=(maxlen,), dtype='int32', name='input_ids')
        attention_mask = Input(shape=(maxlen,), dtype='int32', name='attention_mask')
        
        # 加载RoBERTa模型
        roberta = TFBertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        
        # 根据需要冻结部分层
        label_trainable = False
        if use_att:
            label_trainable = True
            for i in range(6):
                roberta.bert.encoder.layer[i].trainable = False
                
        # 获取RoBERTa输出 [0]表示最后一层隐藏状态 #shape (None, max_len, hidden_size）
        roberta_output = roberta(input_ids, attention_mask=attention_mask)[0]
        
        # 准备标签输入
        count = sum(num_classes_list)
        label_input = Input(shape=(count,), name='label_input')
        
        # 标签嵌入
        if label_emb_matrix is None:
            label_emb = Embedding(
                count, 
                hidden_size, 
                input_length=count, 
                name='label_emb'
            )(label_input)
        else:
            label_emb = Embedding(
                count, 
                hidden_size, 
                input_length=count, 
                weights=[label_emb_matrix], 
                trainable=label_trainable, 
                name='label_emb'
            )(label_input)
        
        # 应用注意力机制
        if use_att:
            idx = 0
            for i in range(hierarchy_levels):
                # 获取当前层次的标签嵌入
                level_label_emb = label_emb[:, idx:idx+num_classes_list[i], :]
                idx += num_classes_list[i]
                
                # 应用注意力
                attention_weight, attention_out, weighted_text_representation = _attention(
                    roberta_output, 
                    level_label_emb, 
                    str(i)+"_attention_layer_"
                )
                
                # 获取[CLS]标记的表示
                roberta_output_pool = roberta_output[:, 0, :]
                
                if i != (hierarchy_levels-1):
                    # 非最后层的处理
                    # 计算权重
                    weight1 = Dense(1, activation='sigmoid')(roberta_output_pool)
                    weight2 = Dense(1, activation='sigmoid')(attention_out)
                    weight1 = weight1 / (weight1 + weight2)
                    weight2 = 1 - weight1
                    
                    # 组合表示
                    doc = weight1 * roberta_output_pool + weight2 * attention_out
                    
                    # 应用局部层
                    local_transmit = _local_layer(
                        doc, 
                        attention_weight, 
                        num_classes_list[i], 
                        str(i) + "_local_layer_"
                    )
                    
                    # 更新roberta输出
                    roberta_output = tf.multiply(
                        roberta_output, 
                        tf.expand_dims(local_transmit, -1)
                    )
                else:
                    # 最后一层使用图注意力
                    batch_size = tf.shape(weighted_text_representation)[0]
                    batch_adj_matrices = tf.tile(
                        tf.expand_dims(labels_relation, 0), 
                        [batch_size, 1, 1]
                    )
                    
                    # 应用图注意力层
                    gat_layer = GraphAttention(
                        F_=hidden_size,
                        attn_heads=8,
                        attn_heads_reduction='average',
                        dropout_rate=0.2,
                        activation='relu'
                    )
                    
                    # 处理图注意力输出
                    gat_output = gat_layer([weighted_text_representation, batch_adj_matrices])
                    gat_attention_out = tf.reduce_mean(gat_output, axis=1)
                    
                    # 计算权重并组合
                    weight3 = Dense(1, activation='sigmoid')(roberta_output_pool)
                    weight4 = Dense(1, activation='sigmoid')(gat_attention_out)
                    weight3 = weight3 / (weight3 + weight4)
                    weight4 = 1 - weight3
                    
                    # 最终输出
                    roberta_output = weight3 * roberta_output_pool + weight4 * gat_attention_out
        else:
            # 不使用注意力时，直接使用[CLS]标记
            roberta_output = roberta_output[:, 0, :]
            
        # 最终预测层
        pred_probs = Dense(
            num_classes_list[-1], 
            activation='sigmoid',
            name='pred_probs'
        )(roberta_output)
        
        # 构建并编译模型
        model = Model(
            inputs=[input_ids, attention_mask, label_input], 
            outputs=pred_probs
        )
        
        optimizer = Adam(learning_rate=2e-5)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer, 
            metrics=metrics
        )
        
        # 修复TensorBoard bug
        model._get_distribution_strategy = lambda: None
        print(model.summary())
        
        return model, roberta_output, label_emb[:, -num_classes_list[-1]:, :]
