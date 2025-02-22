import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Read data
data = pd.read_csv('书籍评价.csv', encoding='gbk')
# print(data)

# 2.1 Take out the content columns and analyze the data
content = data['内容']
# print(content.head())

# 2.2 Judgment Judging Criteria -- 1 favorable review; 0 unfavorable review
data.loc[data.loc[:, '评价'] == "好评", "评论标号"] = 1  # Revise the positive feedback to 1
data.loc[data.loc[:, '评价'] == '差评', '评论标号'] = 0
# print(data.head())

good_or_bad = data['评价'].values
# print(good_or_bad)
# ['好评' '好评' '好评' '好评' '差评' '差评' '差评' '差评' '差评' '好评' '差评' '差评' '差评']

# 2.3 Selection of deactivated words
# Load stop words
stopwords = []
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    # print(lines)
    for tmp in lines:
        line = tmp.strip()
        # print(line)
        stopwords.append(line)
# print(stopwords)
# De-duplication of the list of deactivated words
stopwords = list(set(stopwords))
# print(stopwords)

# 2.4 Processing of “content” into a standardized format
comment_list = []
for tmp in content:
    # print(tmp)
    # Cut text data
    # The cut_all parameter defaults to False, and all cut methods default to exact mode.
    seg_list = jieba.cut(tmp, cut_all=False)
    # print(seg_list)  # <generator object Tokenizer.cut at 0x0000000007CF7DB0>
    seg_str = ','.join(seg_list)  # splice a string
    # print(seg_str)
    comment_list.append(seg_str)  # The purpose is to transform into a list form
# print(comment_list)  # View the comment_list list

# 2.5 Count the number of words
# Perform a count of the number of words
# Instantiate the object
# CountVectorizer class will convert the words in the text into a word frequency matrix
con = CountVectorizer(stop_words=stopwords)
# Word counts
X = con.fit_transform(
    comment_list)  # It calculates the number of occurrences of each word using the fit_transform function
name = con.get_feature_names_out()  # Get the keywords of all texts in the bag of words by get_feature_names().
# print(X.toarray())  # The results of the word frequency matrix can be seen with toarray()
# print(name)

# 2.6 Prepare the training and test sets.
# # Prepare the training set Here, the first 10 lines of the text are used as the training set and the last 3 lines are used as the test set.
x_train = X.toarray()[:10, :]
y_train = good_or_bad[:10]
# Prepare the test sets
x_text = X.toarray()[10:, :]
y_text = good_or_bad[10:]

# Build a Bayesian Algorithm Classifier
mb = MultinomialNB(alpha=1)  # alpha is optional, default 1.0, add Laplacian/Lidstone smoothing parameter.
# Train data
mb.fit(x_train, y_train)
# Predict data
y_predict = mb.predict(x_text)
# Demonstration of predicted and real values
print('Predicted value：', y_predict)
print('Real Value:', y_text)

print(mb.score(x_text, y_text))

"""
Predicted value： ['差评' '差评' '差评']
Real Value: ['差评' '差评' '差评']
1.0
"""