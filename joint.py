from pylab import *
from numpy import *
from PIL import Image
import os

# If you have PCV installed, these imports should work
from PCV.geometry import homography, warp
from PCV.localdescriptors import sift

path='/home/surf/jlb/2222'


def process():
    path1=path+".jpg"
    path2=path+".txt"

    resultname=path2
    params = "--edge-thresh 10 --peak-thresh 5"
    im = Image.open(path1).convert('L')
    im.save('tmp.pgm')
    imagename = 'tmp.pgm'

    cmmd = str("sift " + imagename + "--output=" + resultname +
               " " + params)
    print(cmmd)
    os.system(cmmd)
    print('processed', imagename, 'to', resultname)

    l1, d1 = sift.read_features_from_file(path2)
    im1 = array(Image.open(path1))
    sift.plot_features(im1, l1, circle=False)


    sift.process_image(path1,path2)
    l1, d1 = sift.read_features_from_file(path2)
    im1 = array(Image.open(path1))
    sift.plot_features(im1, l1, circle=False)

if __name__=='__main__':
    process()