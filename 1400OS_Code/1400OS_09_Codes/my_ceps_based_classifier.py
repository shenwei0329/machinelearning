# -*- coding: utf-8 -*-

'''
    2014年12月8日：1:36PM，尚无法理解如何将 测试样本 与 训练样本 进行比较。
                  3:19PM，可以进行分类了。
                  
    
'''

import numpy as np
from collections import defaultdict

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import ShuffleSplit

from sklearn.metrics import confusion_matrix

from utils import plot_roc, plot_confusion_matrix, GENRE_LIST

from ceps import read_ceps

TEST_DIR = "private"

genre_list = GENRE_LIST


def train_model(clf_factory, X, Y, name, plot=False):

    labels = np.unique(Y)

    ''' 获得测试样本
    '''
    test_ceps = np.load('private/v.ceps.npy')
    num_ceps = len(test_ceps)
    test_X = []

    '''抽取测试样本的 中间部分
    '''
    test_X.append(np.mean(test_ceps[int(num_ceps / 10):int(num_ceps * 9 /10)],axis=0))

    print(">>> test_X[%s] <<<" % test_X)

    '''开始处理
           抽取 训练样本
    '''
    
    '''创建 分类器
    '''
    clf = clf_factory()
    '''给分类器设置 训练样本
    '''
    clf.fit(X, Y)

    _my_max = -1
    _my_select = -1

    for _i in labels:
        test_score = clf.score(test_X, [_i])
        print(">>> test_score=[%s],label=[%s]" % (test_score,GENRE_LIST[_i]))
        if test_score > _my_max:
            _my_max = test_score
            _my_select = GENRE_LIST[_i]

    print(">>> My select [%s] <<<" % _my_select)


def create_model():
    from sklearn.linear_model.logistic import LogisticRegression
    '''回归分类器
    '''
    clf = LogisticRegression()

    return clf


if __name__ == "__main__":


    X, y = read_ceps(genre_list)

    train_model(create_model, X, y, "Log Reg CEPS", plot=False)

'''
    Eof
'''
