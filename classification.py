# coding: UTF-8
import os
import numpy as np
import argparse
import logging
from datetime import datetime
import pytz
from sklearn.model_selection import KFold, train_test_split
from utils import build_label_relation_matrix

from models import trainer
import utils

logging.basicConfig(
    level=logging.INFO,
    # filename='train.log',
    # filemode='a',
    format='%(asctime)s : %(levelname)s : %(message)s',
)

parser = argparse.ArgumentParser(description='ATMK')
parser.add_argument('--use_att', default=False, type=bool,
                    help='True for use label attention')
parser.add_argument('--use_lcm', default=False, type=bool,
                    help='True for use label confusion model')
parser.add_argument('--config', default='config/config_a1.yml', type=str,
                    help='config file')
args = parser.parse_args()

if __name__ == '__main__':
    # 加载配置文件
    logging.info("Loading config...")
    config = utils.read_config(args.config)
    logging.info(config)
    # 加载数据
    logging.info("Loading data...")
    label2index,input_ids,attention_mask,label_list = utils.load_data(
        config.cache_file_h5py, config.cache_file_pickle)
    # 
    label_emb_2dlist = None
    if config.get('label_embeddings', None):
        label_emb_2dlist = utils.load_embed_data(config.label_embeddings)
    # 当前模型名称
    model_name = "b"
    if args.use_att & args.use_lcm:
        model_name = "lhabs"
    elif args.use_att:
        model_name = "lhab"
    elif args.use_lcm:
        model_name = "lbs"
    logging.info("model name %s" % model_name)
    # ========== model training: ==========
    X_input_ids = np.array(input_ids)
    X_attention_mask = np.array(attention_mask)
    y = np.array(label_list)
    
    X_train1, X_test, X_train_mask1, X_test_mask, y_train1, y_test = train_test_split(
            X_input_ids, X_attention_mask, y, test_size=0.25, random_state=3407)
    print("TOTAL:", len(X_input_ids),
          "TRAIN:", X_train1, len(X_train1),
          "TEST:", X_test, len(X_test))

    
    '''
    初始化模拟标签数据（L_train,L_test）
    shape=(None,num_classes)
    [[  0   1   2 ... 424 425 426]
        [  0   1   2 ... 424 425 426]
        [  0   1   2 ... 424 425 426]
        ...
        [  0   1   2 ... 424 425 426]
        [  0   1   2 ... 424 425 426]
        [  0   1   2 ... 424 425 426]]
    '''
    label_count = sum(config.num_classes_list[i] for i in range(len(config.num_classes_list)))

    logging.info('=====Start=====')
    kf = KFold(n_splits=5,shuffle=True,random_state=3407)
    fold_results = []

     # 使用完整训练集构建基础关系矩阵
    base_labels_relation = utils.build_label_relation_matrix(
       y_train1, 
       num_classes=config.num_classes_list[-1],
       threshold=config.get('label_relation_threshold', 0.5),
       symmetric=config.get('label_relation_symmetric', False)
    )
    config.labels_relation = base_labels_relation
    print(base_labels_relation)
    
    for k, (train, val) in enumerate(kf.split(X_train1, y_train1)):
        X_train, X_val, X_train_mask,X_val_mask, y_train, y_val = X_train1[train], X_train1[val],X_train_mask1[train],X_train_mask1[val],y_train1[train], y_train1[val]

        L_train = np.array([np.array(range(label_count)) for i in range(len(X_train))])
        L_val = np.array([np.array(range(label_count)) for i in range(len(X_val))])
        L_test = np.array([np.array(range(label_count)) for i in range(len(X_test))])

        print(len(X_train))
        print(len(X_val))
        print(len(X_test))

        file_id = '%s-%s-%s' % (utils.randomword(6), model_name, datetime.now(pytz.timezone('Asia/Shanghai')
                                                                          ).strftime("%m%d-%H%M%S"))
        log_dir = os.path.join('logs', file_id)
        np.random.seed(3407)  # 这样保证了每次试验的seed一致

        labs_model = trainer.LHABSModel(config, label_emb_matrix=label_emb_2dlist, use_att=args.use_att, use_lcm=args.use_lcm, log_dir=log_dir)
        
        labs_model.train_and_val(X_train,X_train_mask, y_train, L_train, X_val, X_val_mask, y_val, L_val)
        result = labs_model.validate(X_test, X_test_mask, y_test, L_test)
        fold_results.append(result)

    print("fold_result: ", fold_results)
    # 计算每个位置的平均值
    average_result = [sum(x)/len(x) for x in zip(*fold_results)]
    print("average_result: ", average_result)

    logging.info('=======End=======')
    # 模型训练完毕后在测试集上最终评估