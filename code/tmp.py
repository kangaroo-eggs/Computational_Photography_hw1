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
from tkinter import image_names
from turtle import color, shape
from unittest import result
from urllib import response
import cv2 as cv
from cv2 import norm
import numpy as np
import matplotlib.pyplot as plt

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
    N, rows, cols = img_samples.shape # N：number of expoture time。rows*cols = number of sample pixel
    
    #build z_ij 從左到右，從上到下
    z = np.zeros((rows*cols, N))
    for n in range(N):   
        z[:,n] = img_samples[n,:,:].flatten('C')    # 'C' means row major
    
    #weighted function
    def w(z):
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
    # (m*n)*k -> m*n
    ''' TODO '''
    N, rows, cols = img_list.shape
    z = np.zeros((rows*cols, N))
    #build z_ij
    for n in range(N): 
        z[:,n] = img_list[n,:,:].flatten('C')  
           
    #weighted function
    def w(z):
        if z <= 0.5*(Z_max+Z_min):
            return z
        else:
            return Z_max-z
    
    #radiance   use one loop to modify
    E, deno, numer = np.zeros((rows*cols)), np.zeros((rows*cols)), np.zeros((rows*cols))
    for i in range(rows*cols):
        w_i, n_i = np.zeros((N)), np.zeros((N))
        for j in range(N):          # 可以連同 w 一起優化? 
            w_i[j] = w(z[i,j])
            n_i[j] = response[int(z[i,j])]
        n_i = n_i - np.log(etime_list)
        numer[i] = np.dot(w_i,n_i)
        deno[i] = np.sum(w_i)     
    E = np.exp(numer/deno)
    radiance = np.reshape(E, (rows,cols))
    
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
    
    #compute B_avg & G_avg & R_avg
    avg = np.zeros((3))
    for i in range(3):
        avg[i] = np.sum(src[i,y_range[0]:y_range[1],x_range[0]:x_range[1]])
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
    N, rows, cols = src.shape
    x_max = np.array([np.max(src[0,:,:])*np.ones((rows,cols)), np.max(src[1,:,:])*np.ones((rows,cols)), np.max(src[2,:,:])*np.ones((rows,cols))])
    X_hat = np.zeros(src.shape)
    for i in range(N):
        X_hat[i,:,:] = scale*(np.log2(src[i,:,:])-np.log2(x_max[i]))+np.log2(x_max[i])
    X_hat = np.exp2(X_hat)

    # gamma correction
    X_prime = np.power(X_hat, 1/gamma)
    
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
    for i in range(N):
        for ele in np.nditer(X_prime[i,:,:], order='C', op_flags=['readwrite']):
            if ele > 1:
                ele[...] = 1    # ele should be indexed with the ellipsis(...)
            elif ele < 0:
                ele[...] = 0
    result = np.around(X_prime*255)  

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
    N, rows, cols = src.shape

    # create intensity map & color ratio
    I = np.zeros((rows,cols))      # intensity map
    for i in range(N):
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
    C = C_bgr
    C*=I_prime

    
    # gamma correction
    C_prime = np.power(C, 1/gamma)
    # clip to range [0,1] -> multiplied by 255 -> rounding
    for i in range(N):
        for ele in np.nditer(C_prime[i,:,:], order='C', op_flags=['readwrite']):    # clip to range[0,1]
            if ele > 1:
                ele[...] = 1
            elif ele < 0:
                ele[...] = 0
    result = np.around(C_prime*255)

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
    extended_src = np.pad(src, ((added, added), (added, added)), 'symmetric')

    # create w_gaussian
    w = np.zeros((2*added+1,2*added+1))     # filter w 的 size 是 odd*odd
    deno = 0; sigma_s = 2*sigma_s**2                                # 分母
    for u in range(w.shape[0]):
        for v in range(w.shape[1]):
            k, l = u-added, v-added
            w[u,v] = -(k**2+l**2)
    w = np.exp(w/sigma_s)
    deno = np.sum(w)
    # print(w)                              # w 的中心是(added, added), 對應座標(0,0) in (k,l) coordinate
    
    # calculate L_B(i,j)
    L_B = np.zeros((src.shape))     #不能直接拿 src 來做!，因為 src 是 mutable 所以會改到原來的 src
    for i in range(rows):
        for j in range(cols):
            L_B[i,j] = np.trace(np.dot(extended_src[i:i+w.shape[0], j:j+w.shape[1]], w.T))   # 分子，draw a picture will easy to understand
    result = L_B/deno

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
    extended_src = np.pad(src, ((added, added), (added, added)), 'symmetric')
    #print('1')
    # calculate L_B(i,j) & w_bilateral
    sigma_s = 2*sigma_s**2; sigma_r = 2*sigma_r**2
    #only calculate the first term onces
    L_B, deno = np.zeros((src.shape)), np.zeros((src.shape))
    for i in range(rows):
        for j in range(cols):
            w, numer_s, numer_r = np.zeros((2*added+1,2*added+1)), np.zeros((2*added+1,2*added+1)), np.zeros((2*added+1,2*added+1))     # filter w
            for u in range(w.shape[0]):
                for v in range(w.shape[1]):
                    k, l = u-added, v-added
                    numer_s[u,v] = -(k**2+l**2)
                    numer_r[u,v] = -(src[i,j]-extended_src[i+added+k, j+added+l])**2
            w = numer_s/sigma_s+numer_r/sigma_r
            w = np.exp(w)
            L_B[i,j] = np.trace(np.dot(extended_src[i:i+w.shape[0], j:j+w.shape[1]], w.T))
            deno[i,j] = np.sum(w)
    result = L_B/deno
    #print('2')
    return result   #L_B

#==============================================================
# 2-1a & 2-2-1
#==============================================================
def Draw_EstimateResponse(img_samples, etime_list, lambda_=50):
    """EstimateResponse & drawing """

    N, rows, cols = img_samples.shape # N：number of expoture time。rows*cols = number of sample pixel
    
    #build z_ij
    z = np.zeros((rows*cols, N))
    for n in range(N):   
        z[:,n] = img_samples[n,:,:].flatten('C')

    def w(z):
        if z <= 0.5*(Z_max+Z_min):
            return z
        else:
            return Z_max-z
    
    A = np.zeros((rows*cols*N+1+254, 256+rows*cols))    # 254 for smoothness term，1 for g(127) = 0；256 for g(0)~g(255)，rows*cols for lnE_i
    # A = np.zeros((rows*cols*N+1, 256+rows*cols))    # without smoothness term
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

    # drawing part
    for x in range(0,rows*cols):    # rows*cols 條線
        plt.plot([np.log(etime_list[i]) for i in range(N)], [z[x][j] for j in range(N)],  'r-', linewidth=1)
    plt.xlabel('log Exposure', fontsize=10, labelpad = 5)
    plt.ylabel('Pixel value', fontsize=10, labelpad = 5)
    plt.show()

    return response

def Draw_ConstructRadiance(img_list, response, etime_list):
    """ Construct radiance map from brackting images

    Args:
        img_list (uint8 ndarray, shape (N, height, width)): N bracketing images (1 channel)
        response (float ndarray, shape (256)): response map
        etime_list (list of float, size N): N exposure times
    
    Returns:
        radiance (float ndarray, shape (height, width)): radiance map
    """
    # (m*n)*k -> m*n
    ''' TODO '''
    N, rows, cols = img_list.shape
    z = np.zeros((rows*cols, N))
    #build z_ij
    for n in range(N): 
        z[:,n] = img_list[n,:,:].flatten('C')  
           
    #weighted function
    def w(z):
        if z <= 0.5*(Z_max+Z_min):
            return z
        else:
            return Z_max-z

    #radiance
    E, deno, numer = np.zeros((rows*cols)), np.zeros((rows*cols)), np.zeros((rows*cols))
    for i in range(rows*cols):
        w_i, n_i = np.zeros((N)), np.zeros((N))
        for j in range(N):          # 可以連同 w 一起優化? 
            w_i[j] = w(z[i,j])
            n_i[j] = response[int(z[i,j])]
        n_i = n_i - np.log(etime_list)
        numer[i] = np.dot(w_i,n_i)
        deno[i] = np.sum(w_i)     
    E = np.exp(numer/deno)
    radiance = np.reshape(E, (rows,cols))
    
    # draw the mixed figure
    # r = 255
    # c = int(np.around(489*np.random.rand()))
    # plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'ro-')
    # c = int(np.around(489*np.random.rand()))
    # plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'go-')
    # c = int(np.around(489*np.random.rand()))
    # plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'bo-')
    # c = int(np.around(489*np.random.rand()))
    # plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'yo-')
    # c = int(np.around(489*np.random.rand()))
    # plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'co-')
    # plt.xlabel('log Exposure', fontsize=10, labelpad = 5)
    # plt.ylabel('Pixel value', fontsize=10, labelpad = 5)
    # plt.show()

    # c = 300
    # r = int(np.around(709*np.random.rand()))
    # plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'ro-')
    # r = int(np.around(709*np.random.rand()))
    # plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'go-')
    # r = int(np.around(709*np.random.rand()))
    # plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'bo-')
    # r = int(np.around(709*np.random.rand()))
    # plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'yo-')
    # r = int(np.around(709*np.random.rand()))
    # plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'co-')
    # plt.xlabel('log Exposure', fontsize=10, labelpad = 5)
    # plt.ylabel('Pixel value', fontsize=10, labelpad = 5)
    # plt.show()

    r = int(np.around(709*np.random.rand()))
    c = int(np.around(489*np.random.rand()))
    plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'ro-')
    r = int(np.around(709*np.random.rand()))
    c = int(np.around(489*np.random.rand()))
    plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'go-')
    r = int(np.around(709*np.random.rand()))
    c = int(np.around(489*np.random.rand()))
    plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'bo-')
    r = int(np.around(709*np.random.rand()))
    c = int(np.around(489*np.random.rand()))
    plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'yo-')
    r = int(np.around(709*np.random.rand()))
    c = int(np.around(489*np.random.rand()))
    plt.plot(response[img_list[:,r,c]], img_list[:,r,c], 'co-')
    plt.xlabel('log Exposure', fontsize=10, labelpad = 5)
    plt.ylabel('Pixel value', fontsize=10, labelpad = 5)
    plt.show()

    return radiance


def Draw_CameraResponseCalibration(src_path, lambda_):
    img_list, exposure_times = LoadExposures(src_path)
    radiance = np.zeros_like(img_list[0], dtype=np.float32)
    pixel_samples = PixelSample(img_list)
    
    for ch in range(3):
        response = Draw_EstimateResponse(pixel_samples[:,ch,:,:], exposure_times, lambda_)   #ch = BGR
        radiance[ch,:,:] = Draw_ConstructRadiance(img_list[:,ch,:,:], response, exposure_times)
        
    return radiance

#======================================================
# Research Study 3
#======================================================
def B_based_WhiteBalance(src, y_range, x_range):    # Python passes mutable objects as references，not call by value
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
    
    #compute B_avg & G_avg & R_avg
    avg = np.zeros((3))
    for i in range(3):
        avg[i] = np.sum(src[i,y_range[0]:y_range[1],x_range[0]:x_range[1]])
    avg /= area

    #compute X prime
    result = src.copy()     # use result = src will fail since src is mutable
    for i in [1,2]:  # G & R
        result[i,:,:] = result[i,:,:]*avg[0]/avg[i]

    return result

def G_based_WhiteBalance(src, y_range, x_range):    # Python passes mutable objects as references，not call by value
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
    
    #compute B_avg & G_avg & R_avg
    avg = np.zeros((3))
    for i in range(3):
        avg[i] = np.sum(src[i,y_range[0]:y_range[1],x_range[0]:x_range[1]])
    avg /= area

    #compute X prime
    result = src.copy()     # use result = src will fail since src is mutable
    #B & R
    for i in range(0,3,2):  # G & R
        print(i)
        result[i,:,:] = result[i,:,:]*avg[1]/avg[i]

    return result