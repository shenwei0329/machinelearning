# -*- coding: utf-8 -*-

'''
    2014年12月8日：1:36PM，尚无法理解如何将 测试样本 与 训练样本 进行比较。
                  3:19PM，可以进行分类了。
                  
    2014年12月9日：
		1）从SRC_DIR中搜索音频文件；
		2）抽取该音频文件的样本；
		3）对样本进行分类；
		4）将音频文件拷贝到DIST_DIR目录下有分类指定的子目录下。

    －目前通过sox命令暂不处理ape文件。（注：ape文件需将其转换为wav文件
    
'''

import os
import sys
import numpy as np

from utils import GENRE_LIST
genre_list = GENRE_LIST

import ceps

TEST_DIR = "private"
DEST_DIR = "/Volumes/TOSHIBA EXT/dstMusic"
SRC_DIR = "/Volumes/TOSHIBA EXT/srcMusic"
TEMP_DIR = "/Users/shenwei/temp"

def getDir(path):
    '''
        功能：获取指定路径下目录列表。
        path：指定路径
        返回：［名称,文件类型］
    '''
    ret = []

    '''获取目录列表
    '''
    for d,fd,fl in os.walk(path):
        if len(fl)>0:
            for _f in fl:
                '''获取文件类型
                '''
                _sufix = os.path.splitext(_f)[1][1:]
                ret.append([os.path.join(d,_f),_sufix])
    return ret

def train_model(clf_factory, X, Y, fn):

    labels = np.unique(Y)

    ''' 获得测试样本
    '''
    test_ceps = np.load(fn)
    num_ceps = len(test_ceps)
    test_X = []

    '''抽取测试样本的 中间部分
    '''
    test_X.append(np.mean(test_ceps[int(num_ceps / 10):int(num_ceps * 9 /10)],axis=0))

    #print(">>> test_X[%s] <<<" % test_X)

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
        #print(">>> test_score=[%s],label=[%s]" % (test_score,GENRE_LIST[_i]))
        if test_score > _my_max:
            _my_max = test_score
            _my_select = GENRE_LIST[_i]

    #print(">>> My select [%s] <<<" % _my_select)
    return _my_select


def create_model():
    from sklearn.linear_model.logistic import LogisticRegression
    '''回归分类器
    '''
    clf = LogisticRegression()
    return clf

def classifi(src_file):

    _fix_len = 22050 * 2 * 180

    flen = os.path.getsize(os.path.join(TEMP_DIR,'temp.wav'))

    _my_dict = {}
    for _name in GENRE_LIST:
        _my_dict[_name] = 0

    if flen>_fix_len:
        os.system('sox "%s/temp.wav" "%s/output.wav" trim 0 60 : newfile : restart' %
            (TEMP_DIR,TEMP_DIR))

        for _i in range(1,(flen/_fix_len+1)):
            wav_fn = '%s/output%03d.wav' % (TEMP_DIR,_i)
            ceps_fn = '%s/output%03d.ceps.npy' % (TEMP_DIR,_i)
            ceps.create_ceps(wav_fn)
            _my_dict[train_model(create_model, X, y, ceps_fn)]+=1

        _max = -1
        _dst = ''
        for _key in _my_dict:
            if _my_dict[_key]>_max:
                _max = _my_dict[_key]
                _dst = _key
    else:
        wav_fn = '%s/temp.wav' % TEMP_DIR
        ceps_fn = '%s/temp.ceps.npy' % TEMP_DIR
        ceps.create_ceps(wav_fn)
        _dst = train_model(create_model, X, y, ceps_fn)

    _cmd = 'mv "%s" "%s/%s"' % (src_file,DEST_DIR,_dst)
    print(">>>[%s]" % _cmd)
    os.system(_cmd)

if __name__ == "__main__":

    if not os.path.isdir(SRC_DIR):
        print("\n\n\tError: %s is not exist\n\n" % SRC_DIR)
        sys.exit()

    '''获取 训练样本
    '''
    X, y = ceps.read_ceps(genre_list)

    if not os.path.isdir(DEST_DIR):
        os.system('mkdir "%s"' % DEST_DIR)
    for _dir in genre_list:
        __dir = DEST_DIR + "/" + _dir
        if not os.path.isdir(__dir):
            __dir = os.path.join(DEST_DIR,_dir)
            _cmd = 'mkdir "%s"' % __dir
            os.system(_cmd)

    d_list = getDir(SRC_DIR)

    for _file in d_list:
        if (_file[1]=='wav') or (_file[1]=='Wav') or (_file[1]=='flac') or (_file[1]=='mp3'):
            cmd = 'sox "%s" -r 22050 -c 1 "%s/temp.wav"' % (_file[0],TEMP_DIR)
            print(">>>cmd:[%s]" % cmd)
            os.system(cmd)
            classifi(_file[0])

'''
    Eof
'''
