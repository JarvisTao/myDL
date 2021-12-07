from __future__ import print_function
import glob
import os
import numpy as np
from tifffile import tifffile

def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines

def get_filenames_whu(filepath, savepath):
    img_list = [i.split('/')[-1] for i in glob.glob('%s/*'%filepath) if os.path.isdir(i)]
    # left_list = [f'{filepath}/{d}/' f for d in img_list]
    left = []
    right = []
    disp = []
    for imgList in img_list:
        left += [img for img in sorted(glob.glob('%s/%s/Left/*.png'%(filepath, imgList)))]
        right += [img for img in sorted(glob.glob('%s/%s/Right/*.png'%(filepath, imgList)))]
        disp += [img for img in sorted(glob.glob('%s/%s/Disparity/*.png'%(filepath, imgList)))]

    with open(savepath,'w') as f:
        for l, r, d in zip(left, right, disp):
            line = 'test' + l.split(filepath)[-1] + ' '*4 + \
                   'test' + r.split(filepath)[-1] + ' '*4 + \
                   'test' + d.split(filepath)[-1] + '\n'
            f.write(line)
            print(line)
    # print(left[:10],len(left))
    # print(right[:10],len(right))
    # print(disp[:10],len(disp))
    #
def get_filenames_igarss(filepath, savepath):
    left = []
    right = []
    disp = []
    # train
    # left += [img for img in sorted(glob.glob('%sTrack2-RGB-ALL/*LEFT_RGB.tif'%(filepath)))]
    # right += [img for img in sorted(glob.glob('%sTrack2-RGB-ALL/*RIGHT_RGB.tif'%(filepath)))]
    # disp += [img for img in sorted(glob.glob('%sTrack2-Truth/*LEFT_DSP.tif'%(filepath)))]

    # val
    left += [img for img in sorted(glob.glob('%sValidate-Track2/*LEFT_RGB.tif'%(filepath)))]
    right += [img for img in sorted(glob.glob('%sValidate-Track2/*RIGHT_RGB.tif'%(filepath)))]
    disp += [img for img in sorted(glob.glob('%sValidate-Track2-Truth/*LEFT_DSP.tif'%(filepath)))]

    # test
    # left += [img for img in sorted(glob.glob('%sValidate-Track2/*LEFT_RGB.tif'%(filepath)))]
    # right += [img for img in sorted(glob.glob('%sValidate-Track2/*RIGHT_RGB.tif'%(filepath)))]
    # disp += [img for img in sorted(glob.glob('%sValidate-Track2-Truth/*LEFT_DSP.tif'%(filepath)))]
    import numpy as np
    with open(savepath,'w') as f:
        for l, r, d in zip(left, right, disp):
            gt = tifffile.imread(d)
            gt[gt==-999.0] = 0
            flag = np.abs(gt.max()) > np.abs(gt.min())

            line = l.split(filepath)[-1] + ' '*4 + \
                   r.split(filepath)[-1] + ' '*4 + \
                   d.split(filepath)[-1] + ' '*4 + \
                   str(flag) + '\n'
            f.write(line)
            print(line)


get_filenames_igarss('/home/jarvis/Research/datasets/igrass/', './filenames/igarss_test.txt')
