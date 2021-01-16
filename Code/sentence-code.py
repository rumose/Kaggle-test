import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
train_filename = 'C:/Users/牟昱辉/Desktop/data/train.tsv'
test_filename = 'C:/Users/牟昱辉/Desktop/data/test.tsv'
print('读数据集')
data_train = pd.read_csv(train_filename, sep='\t', engine='python')
data_test = pd.read_csv(test_filename, sep='\t', engine='python')
data_train.head()
data_train.shape
# 提取训练集中的文本内容
train_sentence = data_train['Phrase']

# 提取测试集中的文本内容
test_sentence = data_test['Phrase']

# 构建一个语料库。通过pandas中的contcat函数将训练集和测试集的文本内容合并到一起
sentences = pd.concat([train_sentence, test_sentence])

# 合并的一起的语料库的规模
sentences.shape
# 提取训练集中的情感标签
label = data_train['Sentiment']

# 导入停词库
stop_words = open(
    'C:/Users/牟昱辉/Desktop/data/stopwords.txt')
# 词袋模型
vectorizer = CountVectorizer(
    analyzer='word',
    ngram_range=(1, 4),
    stop_words=stop_words,
    max_features=15000
)


vectorizer.fit(sentences)
# tf = Tfidfvectorizerr(
#     analyzer = 'word',
#     ngram_range=(1,4),
#     max_features=150000
# )
# tf.fit(sentences)

x_train, x_test, y_train, y_test = train_test_split(
    train_sentence, label, random_state=1234)
# - x_train 训练集数据 （相当于课后习题）
# - x_test 验证集数据 （相当于模拟考试题）
# - y_train 训练集标签 （相当于课后习题答案）
# - y_test 验证集标签（相当于模拟考试题答案）

# 用词袋模型，把训练集和验证集进行特征工程变为向量
x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(x_test)
# 查看训练集中的一个数据
# x_train[1]
lgl = LogisticRegression()
# 训练模型
lgl.fit(x_train, y_train)
# model_sgd = SGDClassifier(loss='modified_huber')
# model_sgd.fit(x_train,  y_train) 
# model_nb = MultinomialNB()
# model_nb.fit(x_train, y_train)

print(
    '词袋方法进行文本特征工程，使用sklearn默认的逻辑回归分类器，验证集上的预测正确率', lgl.score(x_test, y_test))

test_X = vectorizer.transform(data_test['Phrase'])

# 对测试集中的文本，使用lg_final逻辑回归分类器进行预测
predictions = lgl.predict(test_X)

# 查看预测结果
predictions

# 将测试结果加在测试集中
data_test.loc[:, 'Sentiment'] = predictions
data_test.head()
# 整理格式并存储

# loc通过索引标签来抽取数据：
final_data = data_test.loc[:, ['PhraseId', 'Sentiment']]
final_data.head()

# 保存为.csv文件，即为最终结果
final_data.to_csv('C:/Users/牟昱辉/Desktop/data/predict-file-2.csv', index=None)
