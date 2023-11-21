import numpy as np
import cv2

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