''' Functions in HDR flow '''

from ctypes import sizeof
from distutils.command.build import build
from fileinput import close
from math import radians
import os
from pkgutil import extend_path
from re import X
import re
from sqlite3 import Row
from tarfile import LENGTH_NAME
from turtle import shape
from unittest import result
from urllib import response
import cv2 as cv
from cv2 import norm
import numpy as np

Z = 256  # intensity levels
Z_max = 255
Z_min = 0
gamma = 2.2


def ReadImg(path, flag=1):
    img = cv.imread(path, flag)  # flag = 1 means to load a color image
    img = np.transpose(img, (2,0,1))
    return img


def SaveImg(img, path):
    img = np.transpose(img, (1,2,0))
    cv.imwrite(path, img)
    
    
def LoadExposures(source_dir):
    """ load bracketing images folder

    Args:
        source_dir (string): folder path containing bracketing images and a image_list.txt file
                             image_list.txt contains lines of image_file_name, exposure time, ... 
    Returns:
        img_list (uint8 ndarray, shape (N, ch, height, width)): N bracketing images (3 channel)
        exposure_times (list of float, size N): N exposure times
    """
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *_) = line.split()
        filenames += [filename]
        exposure_times += [float(exposure)]
    # filename_string = [os.path.join(source_dir, f) for f in filenames]    
    # print(filename_string)
    img_list = [ReadImg(os.path.join(source_dir, f)) for f in filenames]    #讀出16張 image，each img_list[*] is an image    
    img_list = np.array(img_list)
    # print(img_list.shape)     (16, 3, 710, 490)
    
    return img_list, exposure_times


def PixelSample(img_list):
    """ Sampling

    Args:
        img_list (uint8 ndarray, shape (N, ch, height, width))
        
    Returns:
        sample (uint8 ndarray, shape (N, ch, height_sample_size, width_sample_size))
    """
    # trivial periodic sample
    sample = img_list[:, :, ::64, ::64]
    # print(sample.shape)     (16, 3, 12, 8)
    
    return sample


def EstimateResponse(img_samples, etime_list, lambda_=50):
    """ Estimate camera response for bracketing images

    Args:
        img_samples (uint8 ndarray, shape (N, height_sample_size, width_sample_size)): N bracketing sampled images (1 channel:R or G or B)
        etime_list (list of float, size N): N exposure times
        lambda_ (float): Lagrange multiplier (Defaults to 50)
    
    Returns:
        response (float ndarray, shape (256)): response map
    """
    
    ''' TODO '''
    N, rows, cols = img_samples.shape[0], img_samples.shape[1], img_samples.shape[2] # N：number of expoture time。rows*cols = number of sample pixel
    
    #build z_ij 從左到右，從上到下
    z = np.zeros((rows*cols, N))
    for n in range(N):   
        i = 0
        for r in range(rows):
            for c in range(cols):
                z[i,n] = img_samples[n,r,c]
                i+=1

    # Z_max, Z_min 已被定義過了
    def w(z):
        'weighted function'
        if z <= 0.5*(Z_max+Z_min):
            return z
        else:
            return Z_max-z
    
    A = np.zeros((rows*cols*N+254+1, 256+rows*cols))    # 254 for smoothness term，1 for g(127) = 0；256 for g(0)~g(255)，rows*cols for lnE_i
    B = np.zeros((A.shape[0], 1))
    
    # data term
    k = 0
    for j in range(N):
        for i in range(rows*cols):
            w_ij = w(z[i,j])
            A[k,int(z[i,j])] = w_ij   
            A[k,(256+i)] = -w_ij  
            B[k,0] = w_ij * np.log(etime_list[j])
            k = k+1

    A[k,127] = 1    # for eq: 1*g(127) = 0
    k = k+1          

    # smoothness term
    for i in range(254):
        w_z = w(i+1)
        A[k,i] = lambda_ * w_z
        A[k,i+1] = -2*lambda_ * w_z
        A[k,i+2] = lambda_ * w_z
        k = k+1

    #solve least square
    X = np.linalg.lstsq(A, B, rcond = -1)[0]
    response = np.zeros((256))
    
    for i in range(256):
        response[i] = X[i]

    # (256,) != (256,1)
    # [1,2,3,4] vs [[1],[2],[3],[4]]
    # (256,) != (1,256)
    # [1,2,3,4] vs [[1,2,3,4]]

    # response = X[0:256]
    # print(response.shape)
    # print(type(response))

    return response


def ConstructRadiance(img_list, response, etime_list):
    """ Construct radiance map from brackting images

    Args:
        img_list (uint8 ndarray, shape (N, height, width)): N bracketing images (1 channel)
        response (float ndarray, shape (256)): response map
        etime_list (list of float, size N): N exposure times
    
    Returns:
        radiance (float ndarray, shape (height, width)): radiance map
    """

    ''' TODO '''
    N, rows, cols = img_list.shape[0], img_list.shape[1], img_list.shape[2]
    z = np.zeros((rows*cols, N))
    #build z_ij
    for n in range(N):   
        i = 0
        for r in range(rows):
            for c in range(cols):
                z[i,n] = img_list[n,r,c]
                i+=1
                
    #weighted function
    def w(z):
        'weighted function'
        if z <= 0.5*(Z_max+Z_min):
            return z
        else:
            return Z_max-z

    #radiance
    E = np.zeros((rows,cols))
    for i in range(rows*cols):
        deno = 0; numer = 0        #分母, 分子
        for j in range(N):
            w_ij = w(z[i,j])
            deno += w_ij
            numer += w_ij*(response[int(z[i,j])]-np.log(etime_list[j]))
        r, c = divmod(i, cols)
        E[r, c] = np.exp(numer/deno)
    radiance = E
    
    return radiance


def CameraResponseCalibration(src_path, lambda_):
    img_list, exposure_times = LoadExposures(src_path)
    radiance = np.zeros_like(img_list[0], dtype=np.float32)
    pixel_samples = PixelSample(img_list)
    # print(pixel_samples.shape)
    for ch in range(3):
        response = EstimateResponse(pixel_samples[:,ch,:,:], exposure_times, lambda_)   #ch = BGR
        radiance[ch,:,:] = ConstructRadiance(img_list[:,ch,:,:], response, exposure_times)
        
    return radiance



def WhiteBalance(src, y_range, x_range):    # Python passes mutable objects as references，not call by value
    """ White balance based on Known to be White(KTBW) region

    Args:
        src (float ndarray, shape (ch, height, width)): source radiance BGR
        y_range (tuple of 2 int): location range in y-dimension
        x_range (tuple of 2 int): location range in x-dimension
        
    Returns:
        result (float ndarray, shape (ch, height, width))
    """
   
    ''' TODO '''
    area = (y_range[1]-y_range[0])*(x_range[1]-x_range[0])
    
    #compute B_avg & G_avg & R_avg，可用 np fn??
    avg = np.zeros((3))
    for r in range(*y_range):
        for c in range(*x_range):
            for i in range(3):
                avg[i] += src[i,r,c]
    avg /= area

    #compute X prime
    result = src.copy()     # use result = src will fail since src is mutable
    for i in range(2):  # B & G
        result[i,:,:] = result[i,:,:]*avg[2]/avg[i]

    return result


def GlobalTM(src, scale=1.0):
    """ Global tone mapping

    Args:
        src (float ndarray, shape (ch, height, width)): source radiance image
        scale (float): scaling factor (Defaults to 1.0)
    
    Returns:
        result(uint8 ndarray, shape (ch, height, width)): result HDR image
    """
    
    ''' TODO '''
    rows, cols = src.shape[1], src.shape[2]
    x_max = np.array([np.max(src[0,:,:]), np.max(src[1,:,:]), np.max(src[2,:,:])])
    # print(x_max.shape)
    X_hat = np.zeros((3, rows, cols))
    for i in range(3):
        for r in range(rows):
            for c in range(cols):
                    X_hat[i,r,c] = scale*(np.log2(src[i,r,c])-np.log2(x_max[i]))+np.log2(x_max[i])
    X_hat = np.exp2(X_hat)
    
    # gamma correction
    X_prime = np.power(X_hat, 1/gamma)
    # print(np.max(X_prime[0]), np.max(X_prime[1]), np.max(X_prime[2]))
    
    # clip to range [0,1] -> multiplied by 255 -> rounding (Version 1)
    # x_max = np.array([np.max(X_prime[0]), np.max(X_prime[1]), np.max(X_prime[2])])
    # for i in range(3):
    #     for r in range(rows):
    #         for c in range(cols):
    #             p = np.around(X_prime[i,r,c]/x_max[i]*255)
    #             if p < 0:
    #                 p = 0
    #             X_prime[i,r,c] = p
    #     print(np.max(X_prime[i,:,:]))

    # clip to range [0,1] -> multiplied by 255 -> rounding (Version 2)
    for i in range(3):
        for r in range(rows):
            for c in range(cols):
                if X_prime[i,r,c] > 1:
                    X_prime[i,r,c] = 1
                elif X_prime[i,r,c] < 0:
                    X_prime[i,r,c] = 0
                p = np.around(X_prime[i,r,c]*255)
                X_prime[i,r,c] = p
    result = X_prime

    return result


def LocalTM(src, imgFilter, scale=3.0):
    """ Local tone mapping

    Args:
        src (float ndarray, shape (ch, height, width)): source radiance image
        imgFilter (function): filter function with preset parameters
        scale (float): scaling factor (Defaults to 3.0)
    
    Returns:
        result(uint8 ndarray, shape (ch, height, width)): result HDR image
    """
    
    ''' TODO '''
    rows, cols = src.shape[1], src.shape[2]

    # create intensity map & color ratio
    I = np.zeros((rows,cols))      # intensity map
    for i in range(3):
        I += src[i,:,:]
    I/=3                   #I_ij = (R_ij+G_ij+B_ij)/3

    C_bgr = src.copy()     # color ratio
    C_bgr/=I

    # log of intensity
    L = np.log2(I)

    # separate base layer(L_B) & detail layer(L_D) with filter
    L_B = imgFilter(L)
    L_D = L-L_B

    # Compress the contrast
    L_max, L_min = np.max(L_B), np.min(L_B)
    L_B_prime = (L_B-L_max*np.ones((rows, cols)))*scale/(L_max-L_min)   # scale 2~15 would look good

    # Reconstruct intensity map with adjusted base layer and detail layer,
    I_prime = np.exp2(L_B_prime+L_D)
    
    # Reconstruct color map with adjusted intensity and color ratio
    C = C_bgr.copy()
    C*=I_prime

    
    # gamma correction
    C_prime = np.power(C, 1/gamma)
    # clip to range [0,1] -> multiplied by 255 -> rounding
    for i in range(3):
        for r in range(rows):
            for c in range(cols):
                if C_prime[i,r,c] > 1:
                    C_prime[i,r,c] = 1
                elif C_prime[i,r,c] < 0:
                    C_prime[i,r,c] = 0
                C_prime[i,r,c] = np.around(C_prime[i,r,c]*255)
    result = C_prime
    # print(result)
    return result


def GaussianFilter(src, N=35, sigma_s=100):
    """ Gaussian filter

    Args:
        src (float ndarray, shape (height, width)): source intensity
        N (int): window size of the filter (Defaults to 35)
                 filter indices span [-N/2, N/2]
        sigma_s (float): standard deviation of Gaussian filter (Defaults to 100)
    
    Returns:
        result (float ndarray, shape (height, width))
    """
    
    ''' TODO '''
    rows,cols = src.shape
    added = int(N/2)    # pad 增加的格數
    # print(src.shape)
    extended_src = np.pad(src, ((added, added), (added, added)), 'symmetric')
    # print(extended_src.shape)

    # create w_gaussian
    w = np.zeros((2*added+1,2*added+1))     # filter w 的 size 是 odd*odd
    deno = 0                                # 分母
    for u in range(w.shape[0]):
        for v in range(w.shape[1]):
            k, l = u-added, v-added
            w[u,v] = np.exp(-(k**2+l**2)/(2*sigma_s**2))
            deno += w[u,v]
    # print(w)                              # w 的中心是(added, added), 對應座標(0,0)

    # calculate L_B(i,j)
    L_B = np.zeros((src.shape))     #不能直接拿 src 來做!，因為 src 是 mutable 所以會改到原來的 src
    for i in range(rows):
        for j in range(cols):
            numer = 0                                       # 分子
            for u in range(w.shape[0]):
                for v in range(w.shape[1]):
                    k, l = u-added, v-added                 
                    numer += extended_src[i+added+k, j+added+l]*w[u,v]      # draw a picture is easy to understand
            L_B[i,j] = numer/deno   
    result = L_B
    
    return result   #L_B


def BilateralFilter(src, N=35, sigma_s=100, sigma_r=0.8):
    """ Bilateral filter

    Args:
        src (float ndarray, shape (height, width)): source intensity
        N (int): window size of the filter (Defaults to 35)
                 filter indices span [-N/2, N/2]
        sigma_s (float): spatial standard deviation of bilateral filter (Defaults to 100)
        sigma_r (float): range standard deviation of bilateral filter (Defaults to 0.8)
    
    Returns:
        result (float ndarray, shape (height, width))
    """
    
    ''' TODO '''
    rows,cols = src.shape
    added = int(N/2)    # pad 增加的格數
    # print(src.shape)
    extended_src = np.pad(src, ((added, added), (added, added)), 'symmetric')
    # print(extended_src.shape)

    # calculate L_B(i,j) & w_bilateral
    L_B = np.zeros((src.shape))     #不能直接拿 src 來做!，因為 src 是 mutable 所以會改到原來的 src
    for i in range(rows):
        for j in range(cols):
            w = np.zeros((2*added+1,2*added+1))     # filter w
            deno, numer = 0, 0                                       # 分母, 分子
            for u in range(w.shape[0]):
                for v in range(w.shape[1]):
                    k, l = u-added, v-added    
                    w[u,v] = np.exp(-(k**2+l**2)/(2*sigma_s**2) -(src[i,j]-extended_src[i+added+k, j+added+l])**2/(2*sigma_r**2))
                    deno += w[u,v]            
                    numer += extended_src[i+added+k, j+added+l]*w[u,v]      # draw a picture is easy to understand
            L_B[i,j] = numer/deno
    result = L_B
    
    return result   #L_B