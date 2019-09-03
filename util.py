import sys
import numpy as np
from scipy.sparse import lil_matrix
from konlpy.tag import Okt

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


# Calc Accuracy
def getAcc(Y_pred, Y):
    return (np.array(Y_pred) == np.array(Y)).mean()

# Extract Features
def extract_features(X, word_indices):
    VOCAB_SIZE = len(word_indices)
    features = lil_matrix((len(X), VOCAB_SIZE))

    for iter in range(len(X)):
        for curWord in X[iter][0]:
            index = word_indices.get(curWord)
            if index is not None:
                features[iter, index] += 1

    return features

"""
Req 1-1-2. 토큰화 함수
tokenize(): 텍스트 데이터를 받아 KoNLPy의 okt 형태소 분석기로 토크나이징
    # okt.pos(doc, norm, stem) : norm은 정규화, stem은 근어로 표시하기를 나타냄
"""
def tokenize(doc):
    okt = Okt()
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]