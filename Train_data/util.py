import sys
import numpy as np
import pickle
from scipy.sparse import lil_matrix
from scipy.io import mmwrite, mmread
from konlpy.tag import Okt

"""
Req 1-1-1. 데이터 읽기
read_data(): 데이터를 읽어서 저장하는 함수
"""
def read_data(filename, filetype):
    if filetype == "txt":
        with open(filename, mode='r', encoding='utf-8') as f:
            data = [line.split('\t') for line in f.read().splitlines()]
    elif filetype == "npy":
        data = np.load(filename, allow_pickle=True)
    elif filetype == "pickle":
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    elif filetype == "mtx":
        data = mmread(filename)

    return data

def write_data(filename, filetype, data):
    if filetype == "txt":
        with open(filename, mode='w', encoding='utf-8') as f:
            for i in range(len(data)):
                jlen = len(data[i]) - 1
                for j in range(jlen):
                    f.write(data[i][j] + "\t")
                f.write(data[i][jlen] + "\n")
    elif filetype == "npy":
        np.save(filename, data)
    elif filetype == "pickle":
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    elif filetype == "mtx":
        mmwrite(filename, data)

"""
Req 1-1-2. 토큰화 함수
tokenize(): 텍스트 데이터를 받아 KoNLPy의 okt 형태소 분석기로 토크나이징
    # okt.pos(doc, norm, stem) : norm은 정규화, stem은 근어로 표시하기를 나타냄
"""
def tokenize(doc):
    okt = Okt()
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

# Extract Features
def extract_features(X, word_indices):
    features = lil_matrix((len(X), len(word_indices)))

    for i in range(len(X)):
        for curWord in X[i]:
            index = word_indices.get(curWord)
            if index is not None:
                features[i, index] += 1

    return features

# Calc Accuracy
def getAcc(Y_pred, Y):
    cnt = 0
    for i in range(len(Y)):
        if Y_pred[i] - 1 <= Y[i] and Y_pred[i] + 1 >= Y[i]:
            cnt += 1
    return cnt / len(Y)

################
# Progress Bar #
################
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s __ %s / %s' % (prefix, bar, percent, '%', suffix, iteration, total))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
