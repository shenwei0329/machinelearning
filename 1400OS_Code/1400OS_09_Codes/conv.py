'''
'''

import os
import sys

def convert_wav_file(p_path,path):
    print(">>> [%s/%s] :" % (p_path,path))

    f = os.popen("ls %s/%s"%(p_path,path))
    fs = f.read()
    f.close()
    f_list = fs.split('\n')

    print(">>> [%s]" % f_list)

    for _f in f_list:
        if _f.find('.au'):
            _f_wav = _f.split('.au')[0] + '.wav'
            print('>>> wav file: [%s]' % _f_wav)
	    cmd = 'sox %s/%s/%s wav_dir/%s/%s' % (p_path,path,_f,path,_f_wav)
            print('>>> cmd:[%s]' % cmd)
            os.system(cmd)

def convert_wav(dir_path):

    f = os.popen("ls " + dir_path)
    dirs = f.read()
    f.close()
    dir_list = dirs.split('\n')

    for _dir in dir_list:
        if _dir!='':
            os.system('mkdir wav_dir/%s' % _dir)
            convert_wav_file( dir_path,_dir)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        convert_wav(sys.argv[1])
