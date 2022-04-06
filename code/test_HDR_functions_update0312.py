''' Test functions in HDR flow '''

import unittest
import cv2 as cv
import numpy as np
from functools import partial

from HDR_functions import EstimateResponse, ConstructRadiance, CameraResponseCalibration, \
                          WhiteBalance, GlobalTM, LocalTM, GaussianFilter, BilateralFilter, \
                          ReadImg

### TEST_PAT_SIZE can be assigned to either 'small' or 'large' ###
#-- 'small' stands for small test pattern size
#-- 'large' stands for large test pattern size
# During implementation, it is recommended to set TEST_PAT_SIZE 'small' for quick debugging.
# However, you have to pass the unit test with TEST_PAT_SIZE 'large' to get the full score in each part.
# Note that for large pattern size, the bilateral filtering process may take longer time to complete.
TEST_PAT_SIZE = 'large'

def Cal_PSNR(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse < 6.5025e-6:
        psnr = 100  # set upper bound to avoid divided by zero
    else:
        psnr = 10 * np.log10((255.0 ** 2)/mse)
    
    return psnr

class Test_HDR_functions(unittest.TestCase):

    def test1_EstimateResponse(self):
        img_samples = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/img_samples_1ch.npy')
        etime_list = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/etime_list.npy')
        golden = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/resp_1ch.npy')
        resp_test = EstimateResponse(img_samples, etime_list)
        mse = np.mean((golden - resp_test)**2)
        # print(mse)
        self.assertLessEqual(mse, 0.01)
        return mse

    def test2_ConstructRadiance(self):
        img_list = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/img_list_1ch.npy')
        resp = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/resp_1ch.npy')
        etime_list = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/etime_list.npy')
        golden = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/rad_1ch.npy')
        rad_test = ConstructRadiance(img_list, resp, etime_list)
        mse = np.mean((golden - rad_test)**2)
        # print(mse)
        self.assertLessEqual(mse, 0.01)
        return mse
    
    def test3_WhiteBalance(self):
        src = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/rad.npy')
        y_range = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/y_range.npy')
        x_range = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/x_range.npy')
        golden = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/rad_wb.npy')
        wb_test = WhiteBalance(src, y_range, x_range)
        mse = np.mean((golden - wb_test)**2)
        # print(mse)
        self.assertLessEqual(mse, 0.01)
        return mse

    def test4_GlobalTM(self):
        src = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/rad_wb.npy')
        golden = ReadImg(f'../UnitTestPat_{TEST_PAT_SIZE}/memorial_gtm_scale2.png')
        gtm_test = GlobalTM(src, scale=2.0)
        psnr = Cal_PSNR(golden, gtm_test)
        # print(psnr)
        self.assertGreaterEqual(psnr, 45)
        return psnr

    def test5_Gaussian(self):
        src = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/L.npy')
        golden = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/filter_gau_golden.npy')
        gau_test = GaussianFilter(src, N=15, sigma_s=100)
        mse = np.mean((golden - gau_test)**2)
        # print(mse)
        self.assertLessEqual(mse, 0.01)
        return mse
    
    def test6_LocalTMgaussian(self):
        src = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/rad_wb.npy')
        golden = ReadImg(f'../UnitTestPat_{TEST_PAT_SIZE}/memorial_ltm_gau.png')
        gau = partial(GaussianFilter, N=15, sigma_s=100)
        ltm_gau_test = LocalTM(src, gau, scale=7)
        psnr = Cal_PSNR(golden, ltm_gau_test)
        # print(psnr)
        self.assertGreaterEqual(psnr, 45)
        return psnr
    
    def test7_Bilateral(self):
        src = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/L.npy')
        golden = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/filter_bil_golden.npy')
        bil_test = BilateralFilter(src, N=15, sigma_s=100, sigma_r=0.8)
        mse = np.mean((golden - bil_test)**2)
        # print(mse)
        self.assertLessEqual(mse, 0.01)
        return mse

    def test8_LocalTMbilateral(self):
        src = np.load(f'../UnitTestPat_{TEST_PAT_SIZE}/rad_wb.npy')
        golden = ReadImg(f'../UnitTestPat_{TEST_PAT_SIZE}/memorial_ltm_bil.png')
        bil = partial(BilateralFilter, N=15, sigma_s=100, sigma_r=0.8)
        ltm_bil_test = LocalTM(src, bil, scale=7)
        psnr = Cal_PSNR(golden, ltm_bil_test)
        # print(psnr)
        self.assertGreaterEqual(psnr, 45)
        return psnr

if __name__ == '__main__':
    unittest.main()
