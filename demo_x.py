import os
import pandas as pd
import numpy as np
import random
import jieba 
import xgboost as xgb
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

# xgboost模型初始化设置
data_train=xgb.DMatrix(x_train_weight,label=train_label)
data_val=xgb.DMatrix(x_val_weight, label=val_label)
watchlist = [(data_val, 'eval')]

# booster:
params={'booster':'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'eval_metric': 'merror',
        'max_depth':6,
        'lambda':10,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'min_child_weight':2,
        'eta': 0.025,
        'seed':0,
        'nthread':32,
        'gamma':0.15,
        'learning_rate' : 0.01}

# 建模与预测：50棵树
import time
begin_ = time.time()
xgbt=xgb.train(params,data_train,num_boost_round=50,evals=watchlist,verbose_eval=10)
print("TIme consume: ",time.time()-begin_)
xgbt.save_model('xgb_model.json')
