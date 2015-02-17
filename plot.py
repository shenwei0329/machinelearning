# -*- coding: utf-8 -*-

'''
    为进一步认识基于MFCC的音频（音乐）分类方法，本程序画出训练样本和测试样本图形。
    目的：从图形判断测试样本与训练样本间的吻合度。看与机器的判断是否一致。
'''

import sys
import os
import numpy as np
from matplotlib import pylab

from utils import GENRE_LIST
genre_list = GENRE_LIST

def plotXY(X,Y,T,c):
    '''
        X：训练样本
        Y：训练样本对应的音乐类别
        T：测试样本
        c：归属的类型
    '''
    pylab.clf()

    for _i in range(0,len(X)):
        if Y[_i]==c:
            ''' 画出该类别的训练样本 '''
            pylab.plot(range(0,15),X[_i],':b')

    ''' 画出测试样本 '''
    pylab.plot(range(0,15),T[0],'ok')
    pylab.title(r'<%s>' % genre_list[c])

    ''' 显示图形 '''
    pylab.show()

'''
    EOF
'''
