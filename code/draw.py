'draw Radiometric Calibration'

import cv2 as cv
import numpy as np
from functools import partial
from HDR_functions import CameraResponseCalibration, Draw_CameraResponseCalibration, \
                          WhiteBalance, G_based_WhiteBalance, B_based_WhiteBalance,\
                          GlobalTM, LocalTM, GaussianFilter, BilateralFilter, \
                          ReadImg, SaveImg
##### Test image: memorial #####
TestImage = 'memorial'
print(f'---------- Test Image is {TestImage} ----------')
# Camera response calibration
radiance = Draw_CameraResponseCalibration(f'../TestImage/{TestImage}', lambda_=50)
print('--Camera response calibration done')
# White balance
# ktbw = (419, 443), (389, 401)
# radiance_wb = G_based_WhiteBalance(radiance, *ktbw)     # * is used to unpack，分別傳 (419,443) (389,401)
# print('--White balance done')
# print('--Tone mapping')
# Global tone mapping
# gtm_no_wb = GlobalTM(radiance, scale=1)  # without white balance
# gtm = GlobalTM(radiance_wb, scale=1)     # with white balance
# print('    Global tone mapping done')
# Local tone mapping with gaussian filter
# ltm_filter = partial(GaussianFilter, N=15, sigma_s=100)
# ltm_gaussian = LocalTM(radiance_wb, ltm_filter, scale=7)
# print('    Local tone mapping with gaussian filter done')
# Local tone mapping with bilateral filter
# ltm_filter = partial(BilateralFilter, N=15, sigma_s=100, sigma_r=0.8)
# ltm_bilateral = LocalTM(radiance_wb, ltm_filter, scale=7)
# print('    Local tone mapping with bilateral filter done')
# print('Whole process done\n')
### Save result ###
# print('Saving results...')
# SaveImg(gtm_no_wb, f'../MyHDR_result/Research_Study_3/{TestImage}_gtm_no_wb.png')
# SaveImg(gtm, f'../MyHDR_result/Research_Study_3/{TestImage}_gtm.png')
# SaveImg(ltm_gaussian, f'../MyHDR_result/Research_Study_3/{TestImage}_ltm_gau.png')
# SaveImg(ltm_bilateral, f'../MyHDR_result/Research_Study_3/{TestImage}_ltm_bil.png')
# print('All results are saved\n')