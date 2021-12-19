import os
import pandas as pd
import numpy as np
import random
import jieba 
import lightgbm as lgb
import warnings
import missingno as msno
import seaborn as sns
import warnings
from operator import index
from numpy.lib.function_base import append
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_recall_fscore_support,roc_curve,auc,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

random.seed(2022)
# Debug参数  
DEBUG = True  # Debug模式可快速跑通代码，非Debug模式可得到更好的结果
num_boost_round=100 if DEBUG else 1000
# # 设置迭代次数，默认为100，通常设置为100+
# num_boost_round = 1000

if os.path.exists("../dataset/lgb/jieba_tfidf.npy"):
    tfidf = np.load("../dataset/lgb/jieba_tfidf.npy").tolist()
    train_data = np.load("../dataset/lgb/jieba_train.npy").tolist()
    test_data = np.load("../dataset/lgb/jieba_test.npy").tolist()
    train_label = np.load("../dataset/lgb/jieba_train_label.npy").tolist()
else:
    # 读取训练集和测试集
    train=pd.read_csv("../dataset/lgb/train_data.csv",encoding="'UTF-8'")
    train_data=np.array(train)
    test=pd.read_csv("../dataset/lgb/test_data.csv",encoding="'UTF-8'")
    test_data=np.array(test)
    #用来统计词频
    tfidf=[]
    #去除所有与类别无关的标点符号
    def is_chinese_english(uchar):
        if (uchar >= u'\u4e00' and uchar <= u'\u9fa5') or uchar.isalpha() or uchar.isdigit():
            return True
        else:
            return False

    def process(content):
        ml=map(lambda s:s.replace(' ',''),content)
        str1=''
        for ci in list(ml):
            str1+=ci
        str2=''
        for ci in str1:
            if is_chinese_english(ci):
                str2+=ci
        return str2

    # 去除停用词
    def stopwordslist():
        stopwords = [line.strip() for line in open('../dataset/lgb/stop.txt',encoding='UTF-8').readlines()]
        return stopwords
    stopwords=stopwordslist()

    clean_data=[]
    for item in train_data:
        clean_data.append([item[0],process(item[1])])
    clean_data2=[]
    for item in test_data:
        clean_data2.append([item[0],process(item[1])])

    # 进行jieba分词
    str=[] #分词前
    for item in clean_data:
        str.append([item[0],list(jieba.cut(item[1]))])
    str2=[] #分词前
    for item in clean_data2:
        str2.append([item[0],list(jieba.cut(item[1]))])

    content_clean=[]
    for item in str:
        file=[]
        for ci in item[1]:
            if ci in stopwords:
                continue
            else:
                file.append(ci)
        content_clean.append([item[0],file])

    content_clean2=[]
    for item in str2:
        file=[]
        for ci in item[1]:
            if ci in stopwords:
                continue
            else:
                file.append(ci)
        content_clean2.append([item[0],file])

    # 标签转化，将训练集的标签转化为数字并和内容分离
    train_label=[]
    train_data=[]

    for item in content_clean:
        if item[0]=="财经":
            train_label.append(1)
        elif item[0]=="科技":
            train_label.append(2)
        elif item[0]=="教育":
            train_label.append(3)
        elif item[0]=="家居":
            train_label.append(4)
        elif item[0]=="房产":
            train_label.append(5)
        elif item[0]=="游戏":
            train_label.append(6)
        elif item[0]=="娱乐":
            train_label.append(7)
        elif item[0]=="体育":
            train_label.append(8)
        elif item[0]=="时政":
            train_label.append(9)       
        elif item[0]=="时尚":
            train_label.append(0)  
    # 将jieba分词后的词连接为新的句子
        str=''
        for ci in item[1]:
            str=str+" "+ci
        train_data.append(str)
        tfidf.append(str)

    test_data=[]
    for item in content_clean2:
        str=''
        for ci in item[1]:
            str=str+" "+ci
        test_data.append(str)
        tfidf.append(str)

    i=0
    while i<10:
        i+=1
        print(i,". ")
        print(tfidf[i])
    vector_content=tfidf
    np.save("../dataset/lgb/jieba_tfidf.npy", np.array(tfidf))
    np.save("../dataset/lgb/jieba_train.npy", np.array(train_data))
    np.save("../dataset/lgb/jieba_test.npy", np.array(test_data))
    np.save("../dataset/lgb/jieba_train_label.npy", np.array(train_label))
    
print("Load data done!")

if os.path.exists("../dataset/lgb/tfidf_weight_train.npy"):
    x_train_weight = np.load("../dataset/lgb/tfidf_weight_train.npy")
    x_test_weight = np.load("../dataset/lgb/tfidf_weight_test.npy")
else:
    tfidf_vec = TfidfVectorizer()
    tfidf_vec.fit_transform(vector_content)
    tfidf_matrix1=tfidf_vec.transform(train_data)
    tfidf_matrix2=tfidf_vec.transform(test_data)
    x_train_weight = tfidf_matrix1.toarray()  # 训练集TF-IDF权重矩阵
    x_test_weight = tfidf_matrix2.toarray()  # 测试集TF-IDF权重矩阵
    np.save("../dataset/lgb/tfidf_weight_train.npy", x_train_weight)
    np.save("../dataset/lgb/tfidf_weight_test.npy", x_test_weight)
print("Load tfidf weight done!")
# 创建成lgb特征的数据集格式

val_ratio = 0.1
index_train_val = list(range(len(x_train_weight)))
random.shuffle(index_train_val)
index_val = index_train_val[:int(val_ratio*len(x_train_weight))]
index_train = index_train_val[int(val_ratio*len(x_train_weight)):]

x_val_weight = x_train_weight[index_val, :]
x_train_weight = x_train_weight[index_train, :]
val_label = np.array(train_label)[index_val]
train_label = np.array(train_label)[index_train]

data_train = lgb.Dataset(x_train_weight, train_label, silent=True)
data_val = lgb.Dataset(x_val_weight, val_label, silent=True)
print("length of x_train_weight: {} / length of x_val_weight: {}".format(len(x_train_weight), len(x_val_weight)))
# 构建lightGBM模型
#    参数
params = {
    'boosting_type': 'gbdt', 
    'n_estimators' : 50,
    'objective': 'multiclass',
    'num_threads': 32,
    'num_class': 10,
    'learning_rate': 0.1, 
    'num_leaves': 50, 
    'max_depth': 6,
    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    'metric': 'multi_error',
}
import time
# 训练 lightGBM模型
begin_ = time.time()
if os.path.exists("gbm_model.txt"):
    gbm = lgb.Booster(model_file='gbm_model.txt')
else:
    gbm = lgb.train(params, data_train, num_boost_round, valid_sets=[data_val],verbose_eval=10)
# 预测数据集
# cv_results = lgb.cv(
#              params, data_train, num_boost_round=20, nfold=5, stratified=False, shuffle=True,
#              early_stopping_rounds=100, verbose_eval=4, show_stdv=True, seed=0)

# import pdb;pdb.set_trace()
# exit()

print("Time consume: ", time.time()-begin_)
# y_pred = gbm.predict(x_test_weight, num_iteration=gbm.best_iteration)
# # 输出预测的训练集
# outfile=pd.DataFrame(y_pred)
# outfile.to_csv("result_cls_3.csv",index=False)
# for n in range(955, 1000):
#     graph = lgb.create_tree_digraph(gbm, tree_index=n, name=f'Tree{n}')
#     graph.render(view=False)
gbm.save_model('gbm_model.txt')

