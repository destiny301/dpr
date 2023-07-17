import numpy as np
import torch
import skimage.measure
from torch import nn
import math
import cv2

# color to graysclae images-----------------------------some images are grayscale
def c2g(x):
    # print(x.shape, x.dtype)
    if len(x.shape) == 2:
        HG = x
    elif len(x.shape) == 3 and x.shape[2] == 3:
        x = x/255.0
        x[x<=0.04045] = x[x<=0.04045]/12.92
        x[x>0.04045] = ((x[x>0.04045]+0.055)/1.055)**2.4
        HG = x[:,:, 0]*0.2126 + x[:,:, 1]*0.7152 + x[:,:, 2]* 0.0722
        HG = HG*255.0
        HG = HG.astype(np.uint8)
    else:
        # print(x[:20, :20, 0], x[:20, :20, -1])
        HG = x[:, :, 1]  
    return HG

def padding(x,y):
    h,w,c = x.shape
    size = max(h,w)
    paddingh = (size-h)//2
    paddingw = (size-w)//2
    temp_x = np.zeros((size,size,c))
    temp_y = np.zeros((size,size))
    temp_x[paddingh:h+paddingh,paddingw:w+paddingw,:] = x
    temp_y[paddingh:h+paddingh,paddingw:w+paddingw] = y
    return temp_x,temp_y

def random_crop(x,y, c=None):
    if c is None:
        h,w = y.shape
        randh = np.random.randint(h/8)
        randw = np.random.randint(w/8)
        randf = np.random.randint(10)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth,h+offseth-randh, offsetw, w+offsetw-randw
        if randf >= 5:
            x = x[::, ::-1, ::]
            y = y[::, ::-1]
        return x[p0:p1,p2:p3],y[p0:p1,p2:p3]
    else:
        h,w = y.shape
        randh = np.random.randint(h/8)
        randw = np.random.randint(w/8)
        randf = np.random.randint(10)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth,h+offseth-randh, offsetw, w+offsetw-randw
        if randf >= 5:
            x = x[::, ::-1, ::]
            y = y[::, ::-1]
            c = c[::, ::-1]
        return x[p0:p1,p2:p3],y[p0:p1,p2:p3], c[p0:p1,p2:p3]

def random_rotate(x,y, c=None):
    if c is None:
        angle = np.random.randint(-25,25)
        h, w = y.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(x, M, (w, h)),cv2.warpAffine(y, M, (w, h))
    else:
        angle = np.random.randint(-25,25)
        h, w = y.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(x, M, (w, h)),cv2.warpAffine(y, M, (w, h)), cv2.warpAffine(c, M, (w, h))

def random_light(x):
    contrast = np.random.rand(1)+0.5
    light = np.random.randint(-20,20)
    x = contrast*x + light
    return np.clip(x,0,255)