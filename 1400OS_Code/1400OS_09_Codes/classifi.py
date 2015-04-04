'''
'''

import os
import sys
import fft

def do_file(p_path,path):
    print(">>> [%s/%s] :" % (p_path,path))

    f = os.popen("ls %s/%s"%(p_path,path))
    fs = f.read()
    f.close()
    f_list = fs.split('\n')

    print(">>> [%s]" % f_list)

    for _f in f_list[:-1]:
        if _f.find('.wav')>0:
            print('>>> wav file: [%s]' % _f)
            fn = '%s/%s/%s' % (p_path,path,_f)
            print('--- [%s]' % fn)
            fft.create_fft(fn)

def mk_classifi(dir_path):

    f = os.popen("ls " + dir_path)
    dirs = f.read()
    f.close()
    dir_list = dirs.split('\n')

    for _dir in dir_list[:-1]:
        if _dir!='':
            do_file( dir_path,_dir)

if __name__ == "__main__":
    mk_classifi('wav_dir')
