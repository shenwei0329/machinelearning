# -*- coding: utf-8 -*-

'''

    [[[ Machine Learning 机器学习 ]]]
    
        应用实例： 音频文件分类归档存储
        Created by Shenwei @Chengdu 2014.12.7

    功能：依据训练样本，把指定目录下的所有音频文件（不含ape格式）归类存储到指定的目录下。

    从网上获知，这里采用的mfcc分类是一个不错的算法。

    不足：
        1）目前只能按大的类别归类，如"disco","blues","reggae","jazz","classical", "country", "pop", "rock", "metal"

    改进：
        1）乐器与歌曲，可细分为交响乐、钢琴、独奏与合奏；男声与女声。这需要进一步了解音频特性，需要获得更多的训练样本。
        2）改为GUI方式。从技术上讲，没有难度，只是工作量问题。
        3）更否保留原文件存放位置，只是按分类归档，如：按分类保存源文件的路径。

        注：从互联网可获得很多参考。从中可以了解到，目前对这个感兴趣的人还是很多。分别关注：算法、模型、理论和实践等等。针对音频特性也有
        很多参考，大家对音频信号的分析都很深入，提出的知识也很多。

    2014年12月8日：1:36PM，尚无法理解如何将 测试样本 与 训练样本 进行比较。
                  3:19PM，可以进行分类了。
                  
    2014年12月8日：
		1）从SRC_DIR中搜索音频文件；
		2）抽取该音频文件的样本；
		3）对样本进行分类；
		4）将音频文件拷贝到DIST_DIR目录下有分类指定的子目录下。

        －目前通过sox命令暂不处理ape文件。（注：ape文件需将其转换为wav文件

    2014年12月9日：
                1）优化
                2）添加对 flac 音频文件的归类处理

    2014年12月14日：
                该程序已能正确执行，其分类能力满足要求。

    2014年12月15日：
                为防止目标出现覆盖，在文件中增加“时间戳”。

'''

import sys
import os
import numpy as np
import time

''' 本程序需要的模块 '''
from utils import GENRE_LIST
genre_list = GENRE_LIST
import ceps
import plot

''' 可作为参数输入 '''
DEST_DIR = "/Volumes/TOSHIBA EXT/dstMusic"
SRC_DIR = "/Volumes/TOSHIBA EXT/srcMusic"
TEMP_DIR = "/Users/shenwei/temp"

def getDir(path):
    '''
        功能：获取指定路径下文件列表。
        path：指定路径
        返回：［文件全路径及名称，文件类型（文件后缀）］
    '''
    ret = []

    '''获取目录列表
    '''
    for d,fd,fl in os.walk(path):
        '''
            d：当前目录路径
            fd：当前目录所包含的子目路列表
            fl：当前目录所包含的文件列表
        '''
        if len(fl)>0:
            ''' 若目录下包含有文件，则：'''
            for _f in fl:
                ''' 获取文件类型 '''
                _sufix = os.path.splitext(_f)[1][1:]
                ''' 记录文件（全路径名）及文件后缀'''
                ret.append([os.path.join(d,_f),_sufix])

    ''' 返回指定目录下所有文件的列表 '''
    return ret

def train_model(clf_factory, X, Y, fn):
    ''' 训练模块 '''

    labels = np.unique(Y)

    ''' 装载 测试样本 '''
    test_ceps = np.load(fn)
    num_ceps = len(test_ceps)
    test_X = []

    ''' 抽取测试样本的 中间部分 '''
    test_X.append(np.mean(test_ceps[int(num_ceps / 10):int(num_ceps * 9 /10)],axis=0))

    #print(">>>test_X[%s] <<<" % test_X)

    ''' 创建 分类器 '''
    clf = clf_factory()
    ''' 给分类器设置 训练样本 '''
    clf.fit(X, Y)

    _my_max = -1
    _my_select = -1

    ''' 
    for _i in labels:
        ' 判断各类型匹配得分 '
        test_score = clf.score(test_X, [_i])
        if test_score > _my_max:
            _my_max = test_score
            _my_select = GENRE_LIST[_i]
    '''
    
    ''' 判断测试样本归属于哪个训练样本？ '''
    _i = clf.predict(test_X)
    #print(">>> _i = %d" % _i)

    '''
        显示测试样本在每个 标签 下的相似率
    print clf.predict_proba(test_X)
    '''

    _my_select = GENRE_LIST[_i]
    #print(">>> My select [%s] <<<" % _my_select)

    '''
        显示图形
    for c in labels:
        plot.plotXY(X,Y,test_X,c)
    '''

    return _my_select

def create_model():
    ''' 创建分类器 '''
    from sklearn.linear_model.logistic import LogisticRegression
    ''' 回归分类器 '''
    clf = LogisticRegression()
    return clf

def classifi(src_file):

    ''' 获取音频文件长度 '''
    flen = os.path.getsize(os.path.join(TEMP_DIR,'temp.wav'))
    print(">>> flen=%d" % flen)

    _my_dict = {}
    for _name in GENRE_LIST:
        _my_dict[_name] = 0

    ''' 将大音频文件（大于3分钟的）分割成3分钟采样率22050的单声道音频文件 ''' 
    _fix_len = 22050 * 3 * 60

    n = flen/_fix_len
    print(">>> n=%d" % n)
    
    if flen>_fix_len:
        ''' 将大音频文件拆分成小样本 '''
        os.system(r'sox "%s"/temp.wav "%s"/output.wav trim 0 180 : newfile : restart' %
                  (TEMP_DIR,TEMP_DIR))
                
        wav_fn = '%s/output001.wav' % TEMP_DIR
        flen = os.path.getsize(wav_fn)

        for _i in range(1,n):
            '''对拆分音频进行分类
            '''
            wav_fn = '%s/output%03d.wav' % (TEMP_DIR,_i)
            if flen==os.path.getsize(wav_fn):
                ceps_fn = '%s/output%03d.ceps.npy' % (TEMP_DIR,_i)

                #print(">>> ...[%s]" % ceps_fn)
            
                '''创建测试样本
                '''
                ceps.create_ceps(wav_fn)

                '''样本分类，并分类积分
                '''
                _my_dict[train_model(create_model, X, y, ceps_fn)]+=1

        '''大数判决
        '''
        _max = -1
        _dst= ''
        for _key in _my_dict:
            if _my_dict[_key]>_max:
                _max = _my_dict[_key]
                _dst = _key
    else:
        wav_fn = '%s/temp.wav' % TEMP_DIR
        ceps_fn = '%s/temp.ceps.npy' % TEMP_DIR
            
        '''创建测试样本
        '''
        ceps.create_ceps(wav_fn)

        '''样本分类
        '''
        _dst = train_model(create_model, X, y, ceps_fn)

    ''' 将原始的音频文件 放置 到归类的目录下
        2014.12.14：问题，若在目录下有同名文件，结果？
        －增加时间戳
    '''
    _t = str(time.time())
    _cmd = 'mv "%s" "%s"/%s/"%s"' % (src_file,DEST_DIR,_dst,(_t+'_'+os.path.split(src_file)[1]))
    print(_cmd)
    os.system(_cmd)

if __name__ == "__main__":

    if not os.path.isdir(SRC_DIR):
        print("\n\n\tError: %s is not exist\n\n" % SRC_DIR)
        sys.exit()

    ''' 获取 训练样本 '''
    X, y = ceps.read_ceps(genre_list)

    ''' 建立目标目录 '''
    if not os.path.isdir(DEST_DIR):
        os.system('mkdir "%s"' % DEST_DIR)

    ''' 建立分类子目录 '''
    for _dir in genre_list:
        __dir = os.path.join(DEST_DIR,_dir)
        if not os.path.isdir(__dir):
            _cmd = 'mkdir "%s"' % __dir
            os.system(_cmd)

    d_list = getDir(SRC_DIR)

    for _file in d_list:

        #if (_file[1]=='flac') or (_file[1]=='wav') or (_file[1]=='Wav') or (_file[1]=='mp3'):
        if (_file[1]=='flac') or (_file[1]=='wav') or (_file[1]=='Wav'):
            '''
                对 flac、wav、mp3 格式音频文件分类
                - 将其转换成用于生成测试样本的wav格式文件
            '''
            print(">>> 正在处理[%s]文件" % _file[0])
            _cmd = 'sox "%s" -r 22050 -c 1 "%s"/temp.wav' % (_file[0],TEMP_DIR)
            os.system(_cmd)

            ''' 对该音频文件进行分类存放 '''
            classifi(_file[0])
            
'''
    Eof
'''
