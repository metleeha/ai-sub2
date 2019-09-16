# System & Data 관련 패키지 
import pandas as pd
import pickle
import sys
import os

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

# define 데이터 
OS_PATH = os.path.dirname(__file__) + "/"
OS_DATA_PATH = os.path.dirname(__file__) + "/datafiles/"
# txt 파일
ORIGIN_FILE = "commentsMain.txt"
SPLIT_FILE = "splitLine.txt"
TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"
# npy 파일
TRAIN_NPY_FILE = "train.npy"
TEST_NPY_FILE = "test.npy"
# 최종 파일
WORD_INDICES_FILE = "word_indices.pkl"
MODEL1_FILE = "model1.clf"
MODEL2_FILE = "model2.clf"
MODEL3_FILE = "model3.clf"
MODEL3_LIMIT_FILE = "model3_limit.clf"
# mtx 파일
X_TRAIN_MTX_FILE = "X_train.mtx"
X_TEST_MTX_FILE = "X_test.mtx"
# ovengers 파일
OVENGERS_FILE = "ovengers.clf"
# record별 최대 수
recordCnt = 40000
# train rate
TRAINRATE = 0.8


if os.path.isfile(OS_DATA_PATH + SPLIT_FILE) and \
    os.path.isfile(OS_DATA_PATH + TRAIN_FILE) and \
    os.path.isfile(OS_DATA_PATH + TEST_FILE):
    train_data = np.array(util.read_data(OS_DATA_PATH + TRAIN_FILE, "txt"))
    test_data = np.array(util.read_data(OS_DATA_PATH + TEST_FILE, "txt"))
else:
    origin_data = np.array(util.read_data(OS_DATA_PATH + ORIGIN_FILE, "txt"))

    records = np.zeros(11, dtype=int)
    split_data = []


    for item in origin_data:
        if len(item) >= 3:
            if records[int(item[0])] < recordCnt:
                records[int(item[0])]+=1
                split_data.append(item)

    split_data = np.array(split_data)
    util.write_data(OS_DATA_PATH + SPLIT_FILE, "txt", split_data)

    np.random.shuffle(split_data)
    cnt = int(len(split_data) * TRAINRATE)
    
    train_data = split_data[:cnt]
    test_data = split_data[cnt:]

    util.write_data(OS_DATA_PATH + TRAIN_FILE, "txt", train_data)
    util.write_data(OS_DATA_PATH + TEST_FILE, "txt", test_data)

"""
데이터 전 처리
"""
# 평점 label 데이터가 저장될 Y 행렬 초기화
# Y: train data label
# Y_test: test data label
Y_train = []
Y_test = []
train_review = []
test_review = []

for i in range(len(train_data)):
    if len(train_data[i]) >= 3:
        Y_train.append(int(train_data[i][0]))
        train_review.append(train_data[i][2])

for i in range(len(test_data)):
    if len(test_data[i]) >= 3:
        Y_test.append(int(test_data[i][0]))
        test_review.append(test_data[i][2])

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
train_review = np.array(train_review)
test_review = np.array(test_review)

# Req 1-1-2. 문장 데이터 토큰화
# train_docs, test_docs : 토큰화된 트레이닝, 테스트  문장에 label 정보를 추가한 list
if os.path.isfile(OS_DATA_PATH + TRAIN_NPY_FILE) and \
    os.path.isfile(OS_DATA_PATH + TEST_NPY_FILE):
    train_docs = util.read_data(OS_DATA_PATH + TRAIN_NPY_FILE, "npy")
    test_docs = util.read_data(OS_DATA_PATH + TEST_NPY_FILE, "npy")
else:
    train_docs = []
    test_docs = []

    for i in range(len(train_review)):
        train_docs.append(util.tokenize(train_review[i]))

    for i in range(len(test_review)):
        test_docs.append(util.tokenize(test_review[i]))
    
    train_docs = np.array(train_docs)
    test_docs = np.array(test_docs)

    util.write_data(OS_DATA_PATH + TRAIN_NPY_FILE, "npy", train_docs)
    util.write_data(OS_DATA_PATH + TEST_NPY_FILE, "npy", test_docs)


# Req 1-1-3. word_indices 초기화
word_indices = {}

if os.path.isfile(OS_PATH + "../" + WORD_INDICES_FILE):
    word_indices = util.read_data(OS_PATH + "../" + WORD_INDICES_FILE, "pickle")
else:
    for item in train_docs:
        for word in item:
            if word not in word_indices.keys():
                word_indices[word] = len(word_indices)

    util.write_data(OS_PATH + "../" + WORD_INDICES_FILE, "pickle", word_indices)

# Req 1-1-4. sparse matrix 초기화
# X: train feature data
# X_test: test feature data
if os.path.isfile(OS_DATA_PATH + X_TRAIN_MTX_FILE) and \
    os.path.isfile(OS_DATA_PATH + X_TEST_MTX_FILE):
    X_train = util.read_data(OS_DATA_PATH + X_TRAIN_MTX_FILE, "mtx")
    X_test = util.read_data(OS_DATA_PATH + X_TEST_MTX_FILE, "mtx")
else:
    X_train = util.extract_features(train_docs, word_indices)
    util.write_data(OS_DATA_PATH + X_TRAIN_MTX_FILE, "mtx", X_train)

    X_test = util.extract_features(test_docs, word_indices)
    util.write_data(OS_DATA_PATH + X_TEST_MTX_FILE, "mtx", X_test)

"""
트레이닝 파트
clf  <- Naive baysian mdoel
clf2 <- Logistic regresion model
"""
# Req 1-2-1. Naive baysian mdoel 학습
# Req 1-2-2. Logistic regresion mdoel 학습
# 추가 모델 1 _ Decision Tree
clf = MultinomialNB()
clf2 = LogisticRegression()
# clf3 = DecisionTreeClassifier()
# clf3_limit = DecisionTreeClassifier(max_depth=3)

"""
데이터 저장 파트
"""
# Req 1-4. pickle로 학습된 모델 데이터 저장
if os.path.isfile(OS_PATH + "../" + MODEL1_FILE):
    clf = util.read_data(OS_PATH + "../" + MODEL1_FILE, "pickle")
else:
    clf.fit(X_train, Y_train)
    util.write_data(OS_PATH + "../" + MODEL1_FILE, "pickle", clf)

if os.path.isfile(OS_PATH + "../" + MODEL2_FILE):
    clf2 = util.read_data(OS_PATH + "../" + MODEL2_FILE, "pickle")
else:
    clf2.fit(X_train, Y_train)
    util.write_data(OS_PATH + "../" + MODEL2_FILE, "pickle", clf2)

# if os.path.isfile(OS_PATH + "../" + MODEL3_FILE):
#     clf3 = util.read_data(OS_PATH + "../" + MODEL3_FILE, "pickle")
# else:
#     clf3.fit(X_train, Y_train)
#     util.write_data(OS_PATH + "../" + MODEL3_FILE, "pickle", clf3)

# if os.path.isfile(OS_PATH + "../" + MODEL3_LIMIT_FILE):
#     clf3_limit = util.read_data(OS_PATH + "../" + MODEL3_LIMIT_FILE, "pickle")
# else:
#     clf3_limit.fit(X_train, Y_train)
#     util.write_data(OS_PATH + "../" + MODEL3_LIMIT_FILE, "pickle", clf3_limit)

y_pred = clf.predict(X_test)
y_pred2 = clf2.predict(X_test)
# y_pred3 = clf3.predict(X_test)
# y_pred3_limit = clf3_limit.predict(X_test)

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
# print("Decision Tree classifier accuracy: {}".format(util.getAcc(y_pred3, Y_test)))
# print("Decision Tree_limited classifier accuracy: {}".format(util.getAcc(y_pred3_limit, Y_test)))

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
    def log_likelihoods_naivebayes(self, feature_vector):
        log_likelihood = np.zeros(feature_vector.shape[0])

        smoothing = 1
        smooth_V = feature_vector.shape[0]
        num_feature = np.sum(feature_vector) + smooth_V

        for i in range(smooth_V):
            log_likelihood[i] = np.log(feature_vector[i] + smoothing) - np.log(num_feature)

        return log_likelihood

    """
    Req 3-1-2.
    class_posteriors():
    feature 데이터를 받아 label(class)값에 해당되는 posterior 값들을
    구하고 그 값의 log값을 리턴
    """
    def class_posteriors(self, feature_vector):
        log_likelihood = np.zeros((self.nclass), dtype=float)

        for i in range(1, log_likelihood.shape[0]):
            for feature_index in range(len(feature_vector)):
                log_likelihood[i] += self.likelihoods[i][feature_vector[feature_index]]

        return log_likelihood + self.log_prior

    """
    Req 3-1-3.
    classify():
    feature 데이터에 해당되는 posterir값들(class 개수)을 불러와 비교하여
    더 높은 확률을 갖는 class를 리턴
    """
    def classify(self, feature_vector):
        log_posterior = self.class_posteriors(feature_vector)

        return np.argmax(log_posterior[1:]) + 1

    """
    Req 3-1-4.
    train():
    트레이닝 데이터를 받아 학습하는 함수
    학습 후, 각 class에 해당하는 prior값과 likelihood값을 업데이트
    """
    def train(self, X, Y):
        self.nclass = 11
        nz0 = X.nonzero()[0]
        nz1 = X.nonzero()[1]

        # label(1 ~ 10)에 해당되는 각 feature 성분의 개수값(num_token) 초기화 
        num_token = np.zeros((self.nclass, X.shape[1]), dtype=int)

        for i in range(X.nnz):
            num_token[Y[nz0[i]]][nz1[i]]+=1

        # smoothing을 사용하여 각 클래스에 해당되는 likelihood값 계산
        self.likelihoods = np.zeros((self.nclass, X.shape[1]), dtype=float)
        for i in range(1, self.nclass):
            self.likelihoods[i] = self.log_likelihoods_naivebayes(num_token[i])

        # 각 class의 prior를 계산
        # pior의 log값 계산
        self.log_prior = np.zeros((self.nclass), dtype=float)

        for i in range(1, self.nclass):
            self.log_prior[i] = np.log(num_token[i].sum()) - np.log(X.nnz)

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
model.train(X_train, Y_train)

if not os.path.isfile(OS_PATH + "../" + OVENGERS_FILE):
    with open(OS_PATH + "../" + OVENGERS_FILE, "wb") as f:
        pickle.dump(model, f)

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
        return self.sigmoid(X.dot(beta_x) + beta_c)

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
    def train(self, X, Y):
        # Req 3-3-8. learning rate 조절
        # 학습률(learning rate)를 설정한다.(권장: 1e-3 ~ 1e-6)
        lr = 1
        # 반복 횟수(iteration)를 설정한다.(자연수)
        iters = 1000

        # beta_x, beta_c값을 업데이트 하기 위하여 beta_x_i, beta_c_i값을 초기화
        beta_x_i = np.zeros((X.shape[1], 10), dtype=float)
        beta_c_i = 0
    
        #행렬 계산을 위하여 Y데이터의 사이즈를 (len(Y),1)로 저장합니다.
        Y_hot = np.zeros((X.shape[0], 10), dtype=int)
        for i in range(len(Y)):
            Y_hot[i][Y[i] - 1] = 1
    
        for _ in range(iters):
            #실제 값 Y와 예측 값의 차이를 계산하여 error를 정의합니다.
            error = self.prediction(beta_x_i, beta_c_i, X) - Y_hot

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
        toSig = np.zeros((10), dtype=float)
        for i in X_test:
            for j in range(10):
                toSig[j] += self.beta_x[i][j]
        toSig = self.sigmoid(toSig + self.beta_c)
        return np.argmax(toSig) + 1

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
        return util.getAcc(pred, Y_test)

# Req 3-4-1. model2에 Logistic_Regression_Classifier 클래스를 사용하여 학습합니다.
model2 = Logistic_Regression_Classifier()
model2.train(X_train, Y_train)

# Req 3-4-2. 정확도 측정
print("Logistic_Regression_Classifier accuracy: {}".format(model2.score(X_test, Y_test)))
