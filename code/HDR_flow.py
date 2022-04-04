''' HDR flow '''

import cv2 as cv
import numpy as np
from functools import partial

from HDR_functions import CameraResponseCalibration, WhiteBalance, \
                          GlobalTM, LocalTM, GaussianFilter, BilateralFilter, \
                          ReadImg, SaveImg

##### Test image: racket #####
TestImage = 'racket'
print(f'---------- Test Image is {TestImage} ----------')
### Whole HDR flow ### 
print('Start to process HDR flow...')
# Camera response calibration
radiance = CameraResponseCalibration(f'../my_TestImage/{TestImage}', lambda_=50)
print('--Camera response calibration done')
# White balance
ktbw = (2565, 2655), (1624, 1802)
radiance_wb = WhiteBalance(radiance, *ktbw)     # * is used to unpack，分別傳 (419,443) (389,401)
print('--White balance done')
print('--Tone mapping')
# Global tone mapping
gtm_no_wb = GlobalTM(radiance, scale=1)  # without white balance
gtm = GlobalTM(radiance_wb, scale=1)     # with white balance
print('    Global tone mapping done')
# Local tone mapping with gaussian filter
ltm_filter = partial(GaussianFilter, N=15, sigma_s=100)
ltm_gaussian = LocalTM(radiance_wb, ltm_filter, scale=7)
print('    Local tone mapping with gaussian filter done')
# Local tone mapping with bilateral filter
ltm_filter = partial(BilateralFilter, N=15, sigma_s=100, sigma_r=0.8)
ltm_bilateral = LocalTM(radiance_wb, ltm_filter, scale=7)
print('    Local tone mapping with bilateral filter done')
print('Whole process done\n')
### Save result ###
print('Saving results...')
SaveImg(gtm_no_wb, f'../MyHDR_result/{TestImage}_gtm_no_wb.png')
SaveImg(gtm, f'../mMyHDR_result/{TestImage}_gtm.png')
SaveImg(ltm_gaussian, f'../MyHDR_result/{TestImage}_ltm_gau.png')
SaveImg(ltm_bilateral, f'../MyHDR_result/{TestImage}_ltm_bil.png')
print('All results are saved\n')


# ##### Test image: memorial #####
# TestImage = 'memorial'
# print(f'---------- Test Image is {TestImage} ----------')
# ### Whole HDR flow ### 
# print('Start to process HDR flow...')
# # Camera response calibration
# radiance = CameraResponseCalibration(f'../TestImage/{TestImage}', lambda_=50)
# print('--Camera response calibration done')
# # White balance
# ktbw = (419, 443), (389, 401)
# radiance_wb = WhiteBalance(radiance, *ktbw)     # * is used to unpack，分別傳 (419,443) (389,401)
# print('--White balance done')
# print('--Tone mapping')
# # Global tone mapping
# gtm_no_wb = GlobalTM(radiance, scale=1)  # without white balance
# gtm = GlobalTM(radiance_wb, scale=1)     # with white balance
# print('    Global tone mapping done')
# # Local tone mapping with gaussian filter
# ltm_filter = partial(GaussianFilter, N=15, sigma_s=100)
# ltm_gaussian = LocalTM(radiance_wb, ltm_filter, scale=7)
# print('    Local tone mapping with gaussian filter done')
# # Local tone mapping with bilateral filter
# ltm_filter = partial(BilateralFilter, N=15, sigma_s=100, sigma_r=0.8)
# ltm_bilateral = LocalTM(radiance_wb, ltm_filter, scale=7)
# print('    Local tone mapping with bilateral filter done')
# print('Whole process done\n')
# ### Save result ###
# print('Saving results...')
# SaveImg(gtm_no_wb, f'../Result/{TestImage}_gtm_no_wb.png')
# SaveImg(gtm, f'../Result/{TestImage}_gtm.png')
# SaveImg(ltm_gaussian, f'../Result/{TestImage}_ltm_gau.png')
# SaveImg(ltm_bilateral, f'../Result/{TestImage}_ltm_bil.png')
# print('All results are saved\n')


# ##### Test image: vinesunset #####
# TestImage = 'vinesunset'
# print(f'---------- Test Image is {TestImage} ----------')
# print('Start to process HDR flow...')
# ### Note: File vinesunset.hdr is a raster image or digital photo saved in High Dynamic Range (HDR) image format,
# ###       so we don't perform camera response calibration here.
# ###       Also, we don't perform white balance here because the color of the white region in the result seems normal.
# radiance_vinesunset = ReadImg(f'../TestImage/{TestImage}.hdr', -1)
# print('--Tone mapping')
# # Global tone mapping
# gtm = GlobalTM(radiance_vinesunset, scale=3)
# print('    Global tone mapping done')
# # Local tone mapping with gaussian filter
# ltm_filter = partial(GaussianFilter, N=35, sigma_s=100)
# ltm_gaussian = LocalTM(radiance_vinesunset, ltm_filter, scale=3)
# print('    Local tone mapping with gaussian filter done')
# # Local tone mapping with bilateral filter
# ltm_filter = partial(BilateralFilter, N=35, sigma_s=100, sigma_r=0.8)
# ltm_bilateral = LocalTM(radiance_vinesunset, ltm_filter, scale=3)
# print('    Local tone mapping with bilateral filter done')
# print('Whole process done\n')

# ### Save result ###
# print('Saving results...')
# SaveImg(gtm, f'../Result/{TestImage}_gtm.png')
# SaveImg(ltm_gaussian, f'../Result/{TestImage}_ltm_gau.png')
# SaveImg(ltm_bilateral, f'../Result/{TestImage}_ltm_bil.png')
# print('All results are saved\n')