# System & Data 관련 패키지 
import pandas as pd
import pickle
import json
import sys
import os
from scipy.io import mmwrite, mmread

# 수학 관련 
import numpy as np
import math

# 한글 형태소 분리 
from konlpy.tag import Okt

# ML 모델
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 출력 & Customize
from pprint import pprint
import util
from util import extract_features, tokenize

# define 데이터 
OS_PATH = os.path.dirname(__file__)
OS_DATA_PATH = os.path.dirname(__file__) + "/datafiles"

"""
Req 1-1-1. 데이터 읽기
read_data(): 데이터를 읽어서 저장하는 함수
"""
def read_data(filename):
    with open(OS_PATH + "/" + filename, mode='r', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

"""
데이터 전 처리
"""
# Req 1-1-2. 문장 데이터 토큰화
# train_docs, test_docs : 토큰화된 트레이닝, 테스트  문장에 label 정보를 추가한 list
if os.path.isfile(OS_PATH + '/train_docs.json'):
    with open(OS_PATH + '/train_docs.json', encoding="utf-8") as f:
        train_docs = json.load(f)
    with open(OS_PATH + '/test_docs.json', encoding="utf-8") as f:
        test_docs = json.load(f)
else:
    # train, test 데이터 읽기
    train_data = read_data('ratings_train.txt')
    test_data = read_data('ratings_test.txt')

    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
    # JSON 파일로 저장
    with open(OS_PATH + '/train_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    with open(OS_PATH + '/test_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")

# Req 1-1-3. word_indices 초기화
word_indices = {}

if os.path.isfile('word_indices.pkl'):
    with open('word_indices.pkl', 'rb') as f:
        word_indices = pickle.load(f)
else:
    # Req 1-1-3. word_indices 채우기
    for item in train_docs:
        for word in item[0]:
            if word not in word_indices.keys():
                word_indices[word] = len(word_indices)

    with open('word_indices.pkl', 'wb') as f:
        pickle.dump(word_indices, f, pickle.HIGHEST_PROTOCOL)


# 평점 label 데이터가 저장될 Y 행렬 초기화
# Y: train data label
# Y_test: test data label
Y = np.asarray(np.array(train_docs).T[1], dtype=int)
Y_test = np.asarray(np.array(test_docs).T[1], dtype=int)

"""
트레이닝 파트
clf  <- Naive baysian mdoel
clf2 <- Logistic regresion model
"""

# Req 1-1-4. sparse matrix 초기화
# X: train feature data
# X_test: test feature data
if os.path.isfile('X_train.mtx'):
    X_train = mmread('X_train.mtx')
else:
    X_train = extract_features(train_docs, word_indices)
    mmwrite('X_train.mtx', X_train)

if os.path.isfile('X_test.mtx'):
    X_test = mmread('X_test.mtx')
else:
    X_test = extract_features(test_docs, word_indices)
    mmwrite('X_test.mtx', X_test)


# Req 1-2-1. Naive baysian mdoel 학습
# Req 1-2-2. Logistic regresion mdoel 학습
# 추가 모델 1 _ Decision Tree
clf = MultinomialNB()
clf2 = LogisticRegression()
clf3 = DecisionTreeClassifier()
clf3_limit = DecisionTreeClassifier(max_depth=3)

"""
데이터 저장 파트
"""
# Req 1-4. pickle로 학습된 모델 데이터 저장
if os.path.isfile("model1.clf"):
    with open("model1.clf", "rb") as f:
        clf = pickle.load(f)
else:
    clf.fit(X_train, Y)
    with open("model1.clf", "wb") as f:
        pickle.dump(clf, f)

if os.path.isfile("model2.clf"):
    with open("model2.clf", "rb") as f:
        clf2 = pickle.load(f)
else:
    clf2.fit(X_train, Y)
    with open("model2.clf", "wb") as f:
        pickle.dump(clf2, f)

if os.path.isfile("model3.clf"):
    with open("model3.clf", "rb") as f:
        clf3 = pickle.load(f)
else:
    clf3.fit(X_train, Y)
    with open("model3.clf", "wb") as f:
        pickle.dump(clf3, f)

if os.path.isfile("model3_limit.clf"):
    with open("model3_limit.clf", "rb") as f:
        clf3_limit = pickle.load(f)
else:
    clf3_limit.fit(X_train, Y)
    with open("moel3_limit.clf", "wb") as f:
        pickle.dump(clf3_limit, f)

y_pred = clf.predict(X_test)
y_pred2 = clf2.predict(X_test)
y_pred3 = clf3.predict(X_test)
y_pred3_limit = clf3_limit.predict(X_test)

def getAcc(Y_pred, Y):
    return (np.array(Y_pred) == np.array(Y)).mean()

"""
테스트 파트
"""
# Req 1-3-1. 문장 데이터에 따른 예측된 분류값 출력
print("Naive bayesian classifier example result: {}, {}".format(Y_test[1], y_pred[1]))
print("Logistic regression exampleresult: {}, {}".format(Y_test[1], y_pred2[1]))

# Req 1-3-2. 정확도 출력
print("Naive bayesian classifier accuracy: {}".format(util.getAcc(y_pred, Y_test)))
print("Logistic regression accuracy: {}".format(util.getAcc(y_pred2, Y_test)))

#############################################
# 추가된 모델 정확도 출력
#############################################
print("Decision Tree classifier accuracy: {}".format(util.getAcc(y_pred3, Y_test)))
print("Decision Tree_limited classifier accuracy: {}".format(util.getAcc(y_pred3_limit, Y_test)))

# Naive bayes classifier algorithm part
# 아래의 코드는 심화 과정이기에 사용하지 않는다면 주석 처리하고 실행합니다.

"""
Naive_Bayes_Classifier 알고리즘 클래스입니다.
"""
class Naive_Bayes_Classifier(object):
    """
    Req 3-1-1.
    log_likelihoods_naivebayes():
    feature 데이터를 받아 label(class)값에 해당되는 likelihood 값들을
    naive한 방식으로 구하고 그 값의 log값을 리턴
    """
    def log_likelihoods_naivebayes(self, feature_vector, Class):
        log_likelihood = np.zeros(len(feature_vector))
        smoothing = 1

        if Class == 0:
            smooth_V = len(feature_vector)
            num_feature = np.sum(feature_vector) + smooth_V

            for feature_index in range(len(feature_vector)):
                if feature_vector[feature_index] > 0: #feature present
                    log_likelihood[feature_index] += np.log((feature_vector[feature_index]+smoothing) / num_feature)
                elif feature_vector[feature_index] == 0: #feature absent
                    log_likelihood[feature_index] += np.log(1 / num_feature)
        elif Class == 1:
            smooth_V = len(feature_vector)
            num_feature = np.sum(feature_vector) + smooth_V

            for feature_index in range(len(feature_vector)):
                if feature_vector[feature_index] > 0:
                    log_likelihood[feature_index] += np.log((feature_vector[feature_index]+smoothing) / num_feature)
                elif feature_vector[feature_index] == 0:
                    log_likelihood[feature_index] += np.log(1 / num_feature)
        return log_likelihood

    """
    Req 3-1-2.
    class_posteriors():
    feature 데이터를 받아 label(class)값에 해당되는 posterior 값들을
    구하고 그 값의 log값을 리턴
    """
    def class_posteriors(self, feature_vector):
        # log_likelihood_0 = self.log_likelihoods_naivebayes(feature_vector, Class=0)
        # log_likelihood_1 = self.log_likelihoods_naivebayes(feature_vector, Class=1)
        # 여기서 log_likelihoods를 메소드에 들어가서 가져오지 말고, 이미 학습할 때 구해놓은 값을 이용해서!
        log_likelihood_0 = 0.0
        log_likelihood_1 = 0.0
        for feature_index in range(len(feature_vector)):
            log_likelihood_0 += self.likelihoods_0[feature_vector[feature_index]]
            log_likelihood_1 += self.likelihoods_1[feature_vector[feature_index]]

        log_posterior_0 = log_likelihood_0 + self.log_prior_0
        log_posterior_1 = log_likelihood_1 + self.log_prior_1

        return log_posterior_0, log_posterior_1

    """
    Req 3-1-3.
    classify():
    feature 데이터에 해당되는 posterir값들(class 개수)을 불러와 비교하여
    더 높은 확률을 갖는 class를 리턴
    """

    def classify(self, feature_vector):
        log_posterior_0, log_posterior_1 = self.class_posteriors(feature_vector)
        if log_posterior_0 > log_posterior_1:
            return 0
        else:
            return 1

    """
    Req 3-1-4.
    train():
    트레이닝 데이터를 받아 학습하는 함수
    학습 후, 각 class에 해당하는 prior값과 likelihood값을 업데이트
    """
    def train(self, X, Y):

        # label 0에 해당되는 각 feature 성분의 개수값(num_token_0) 초기화 
        num_token_0 = np.zeros((1, X.shape[1]))
        # label 1에 해당되는 각 feature 성분의 개수값(num_token_1) 초기화
        num_token_1 = np.zeros((1, X.shape[1]))

        # 데이터의 num_0,num_1,num_token_0,num_token_1 값 계산
        if os.path.isfile('num_token_1.npy'):
            num_token_0 = np.load('num_token_0.npy')
            num_token_1 = np.load('num_token_1.npy')
        else:
            # JSON 파일로 저장
            for i in range(X.shape[0]):
                if Y[i] == 0:
                    num_token_0 += X.getrow(i)
                elif Y[i] == 1:
                    num_token_1 += X.getrow(i)

            np.save('num_token_0.npy', num_token_0)
            np.save('num_token_1.npy', num_token_1)

        # smoothing을 사용하여 각 클래스에 해당되는 likelihood값 계산
        self.likelihoods_0 = self.log_likelihoods_naivebayes(num_token_0[0], 0)
        self.likelihoods_1 = self.log_likelihoods_naivebayes(num_token_1[0], 1)

        # 각 class의 prior를 계산
        prior_probability_0 = np.mean(Y == 0)
        prior_probability_1 = np.mean(Y == 1)

        # pior의 log값 계
        self.log_prior_0 = np.log(prior_probability_0)
        self.log_prior_1 = np.log(prior_probability_1)

    """
    Req 3-1-5.
    predict():
    테스트 데이터에 대해서 예측 label값을 출력해주는 함수
    """
    def predict(self, X):
        nonzeros0 = X.nonzero()[0]
        nonzeros1 = X.nonzero()[1]
        predictions = []
        X_test = []
        for i in range(X.shape[0]):
            tmp = []
            X_test.append(tmp)

        for i in range(X.nnz):
            X_test[nonzeros0[i]].append(nonzeros1[i])

        X_test = np.array(X_test)
        
        for case in X_test:
            predictions.append(self.classify(case))

        return predictions


    """
    Req 3-1-6.
    score():
    테스트 데이터를 받아 예측된 데이터(predict 함수)와
    테스트 데이터의 label값을 비교하여 정확도를 계산
    """
    def score(self, X_test, Y_test):
        pred = self.predict(X_test)
        return util.getAcc(pred, Y_test)
        
# Req 3-2-1. model에 Naive_Bayes_Classifier 클래스를 사용하여 학습합니다.
model = Naive_Bayes_Classifier()
model.train(X_train, Y)

# Req 3-2-2. 정확도 측정
print("Naive_Bayes_Classifier accuracy: {}".format(model.score(X_test, Y_test)))

# Logistic regression algorithm part
# 아래의 코드는 심화 과정이기에 사용하지 않는다면 주석 처리하고 실행합니다.

"""
Logistic_Regression_Classifier 알고리즘 클래스입니다.
"""
class Logistic_Regression_Classifier(object):

    """
    Req 3-3-1.
    sigmoid():
    인풋값의 sigmoid 함수 값을 리턴
    """
    def sigmoid(self,z):
        return 1. / (1 + np.exp(-z))

    """
    Req 3-3-2.
    prediction():
    X 데이터와 beta값들을 받아서 예측 확률P(class=1)을 계산.
    X 행렬의 크기와 beta의 행렬 크기를 맞추어 계산.
    ex) sigmoid(            X           x(행렬곱)       beta_x.T    +   beta_c)       
                (데이터 수, feature 수)             (feature 수, 1)
    """

    def prediction(self, beta_x, beta_c, X):
        # 예측 확률 P(class=1)을 계산하는 식을 만든다.
        return self.sigmoid(X * beta_x + beta_c)

    """
    Req 3-3-3.
    gradient_beta():
    beta값에 해당되는 gradient값을 계산하고 learning rate를 곱하여 출력.
    """
    def gradient_beta(self, X, error, lr):
        # beta_x를 업데이트하는 규칙을 정의한다.
        beta_x_delta = X.T * error * lr / X.shape[0]
        # beta_c를 업데이트하는 규칙을 정의한다.
        beta_c_delta = np.mean(error) * lr
        
        return beta_x_delta, beta_c_delta

    """
    Req 3-3-4.
    train():
    Logistic Regression 학습을 위한 함수.
    학습데이터를 받아서 최적의 sigmoid 함수으로 근사하는 가중치 값을 리턴.

    알고리즘 구성
    1) 가중치 값인 beta_x_i, beta_c_i 초기화
    2) Y label 데이터 reshape
    3) 가중치 업데이트 과정 (iters번 반복) 
    3-1) prediction 함수를 사용하여 error 계산
    3-2) gadient_beta 함수를 사용하여 가중치 값 업데이트
    4) 최적화 된 가중치 값들 리턴
       self.beta_x, self.beta_c
    """
    def loss(self, beta_x_i, beta_c_i, X, Y):
        return self.prediction(beta_x_i, beta_c_i, X) - Y


    def train(self, X, Y):
        # Req 3-3-8. learning rate 조절
        # 학습률(learning rate)를 설정한다.(권장: 1e-3 ~ 1e-6)
        lr = 1
        # 반복 횟수(iteration)를 설정한다.(자연수)
        iters = 1000
        
        # beta_x, beta_c값을 업데이트 하기 위하여 beta_x_i, beta_c_i값을 초기화
        beta_x_i = np.zeros((X.shape[1], 1), dtype=float)
        beta_c_i = 0
    
        #행렬 계산을 위하여 Y데이터의 사이즈를 (len(Y),1)로 저장합니다.
        Y = Y.reshape(len(Y), 1)
    
        for _ in range(iters):
            #실제 값 Y와 예측 값의 차이를 계산하여 error를 정의합니다.
            error = self.prediction(beta_x_i, beta_c_i, X) - Y

            #gredient_beta함수를 통하여 델타값들을 업데이트 합니다.
            beta_x_delta, beta_c_delta = self.gradient_beta(X, error, lr)
            beta_x_i -= beta_x_delta
            beta_c_i -= beta_c_delta
            
        self.beta_x = beta_x_i
        self.beta_c = beta_c_i
        
    """
    Req 3-3-5.
    classify():
    확률값을 0.5 기준으로 큰 값은 1, 작은 값은 0으로 리턴
    """
    def classify(self, X_test):
        toSig = 0
        for i in X_test:
            toSig += self.beta_x[i]

        re = self.sigmoid(toSig + self.beta_c)
        if re >= 0.5:
            return 1
        else:
            return 0

    """
    Req 3-3-6.
    predict():
    테스트 데이터에 대해서 예측 label값을 출력해주는 함수
    """
    def predict(self, X):
        nonzeros0 = X.nonzero()[0]
        nonzeros1 = X.nonzero()[1]
        predictions = []

        X_test = []
        for i in range(X.shape[0]):
            tmp = []
            X_test.append(tmp)

        for i in range(X.nnz):
            X_test[nonzeros0[i]].append(nonzeros1[i])

        X_test = np.array(X_test)
        
        for case in X_test:
            predictions.append(self.classify(case))

        return predictions


    """
    Req 3-3-7.
    score():
    테스트를 데이터를 받아 예측된 데이터(predict 함수)와
    테스트 데이터의 label값을 비교하여 정확도를 계산
    """
    def score(self, X_test, Y_test):
        pred = self.predict(X_test)
        return (np.array(pred) == np.array(Y_test)).mean()

# Req 3-4-1. model2에 Logistic_Regression_Classifier 클래스를 사용하여 학습합니다.
model2 = Logistic_Regression_Classifier()
model2.train(X_train, Y)

# Req 3-4-2. 정확도 측정
print("Logistic_Regression_Classifier accuracy: {}".format(model2.score(X_test, Y_test)))

