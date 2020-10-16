# coding=utf-8
import os 
import re 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
train_filename = 'C:/Users/牟昱辉/Desktop/mid-test/train.json'
test_filename = 'C:/Users/牟昱辉/Desktop/mid-test/test.json'

print('读取训练集')
train = pd.read_json(train_filename)
# 最大特征数量 词频排序的前max_features个词创建语料库（降序排列）
vectorizer = CountVectorizer(max_features=2000)
# 训练集菜单
ingredients = train['ingredients']
# print ingredients 
# 原料中用“，”这里换成空格隔开-构造词袋 
words_list = [' '.join(x) for x in ingredients]
print(len(words_list)) 

print('构造词袋')
# 各个词出现的次数 并转换成矩阵的形式 
bag_of_words = vectorizer.fit(words_list)
# 将列表转为数组
bag_of_words = vectorizer.transform(words_list).toarray()
print(bag_of_words.shape)
# print bag_of_words 

# 测试数据集处理
print('read test')
test = pd.read_json(test_filename)
# test.head()
print('test')
test_ingredients = test['ingredients']
test_ingredients_words = [' '.join(x) for x in test_ingredients]
# 要求就是这里只使用transfrom方法
test_ingredients_array = vectorizer.transform(test_ingredients_words).toarray()


# 随机森林
print("随机森林")
forest = RandomForestClassifier(n_estimators=200)
# 训练随机森林
forest = forest.fit(bag_of_words, train['cuisine'])
# 开始预测
print('predict')
result = forest.predict(test_ingredients_array)
output = pd.DataFrame(data={"id": test["id"], "cuisine": result})
print('save')
result_filename = 'C:/Users/牟昱辉/Desktop/mid-test/predict-file.csv'
output.to_csv(result_filename, index=False, quoting=3)
print(result)
