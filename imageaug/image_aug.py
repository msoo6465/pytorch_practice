import cv2
import os
import numpy as np
from imgaug import augmenters as iaa
from turtle import *
from random import randint
import matplotlib.pyplot as plt
from scipy import fftpack

def make_moire_pattern():
    x = np.arange(500) / 500 - 0.5
    y = np.arange(500) / 500 - 0.5

    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    f0 = 10
    k = 250
    a = np.sin(np.pi * 2 * (f0 * R + k * R ** 2 / 2))
    return a

# moire = make_moire_pattern()

img_path = 'C:/dataset/testworks/val (2)/img/eavesdrop/day/20man/villa/eaves_villa_spring_day_01/'
noise_img ='C:/dataset/testworks/noise_image/img_150.png'


# input_img = cv2.imread(img_path)
# gray_img = cv2.imread(img_path,0)
# noise_img = cv2.imread(noise_img)

# print(input_img.shape)
# r = noise_img[:,:,0]
# g = noise_img[:,:,1]
# b = noise_img[:,:,2]
#
# matlist = []
# f1_r = fftpack.fft2((r).astype(float))
# f2_r = fftpack.fftshift(f1_r)
# matlist.append(f2_r)
# f1_g = fftpack.fft2((g).astype(float))
# f2_g = fftpack.fftshift(f1_g)
# matlist.append(f2_g)
# f1_b = fftpack.fft2((b).astype(float))
# f2_b = fftpack.fftshift(f1_b)
# matlist.append(f2_b)
#
# def norm_fft(mat):
#     return (20 * np.log10(0.1 + mat)).astype(int)
#
# for i in range(3):
#     plt.subplot(2,3,i+1),plt.imshow(norm_fft(matlist[i]),cmap=plt.cm.gray)
# # plt.show()
#
# r = input_img[:,:,0]
# g = input_img[:,:,1]
# b = input_img[:,:,2]
#
# r = cv2.filter2D(r,-1,norm_fft(f2_r))
# g = cv2.filter2D(g,-1,norm_fft(f2_g))
# b = cv2.filter2D(b,-1,norm_fft(f2_b))
#
# r = np.reshape(r,(720,1280,1))
# g = np.reshape(g,(720,1280,1))
# b = np.reshape(b,(720,1280,1))
#
# t = np.concatenate((r,g),axis=2)
# t = np.concatenate((t,b),axis=2)
# cv2.imshow('sd',t)
# cv2.waitKey(0)
# exit()
#
# im = cv2.imread(img_path)
# # im = cv2.medianBlur(im,7)
# F1 = fftpack.fft2((im).astype(float))
# F2 = fftpack.fftshift(F1)
# w,h = im.shape
# for i in range(60, w, 135):
#     for j in range(100, h, 200):
#         if not (i == 330 and j == 500):
#             F2[i-20:i+20, j-20:j+30] = 0
# for i in range(0, w, 135):
#     for j in range(200, h, 200):
#         if not (i == 330 and j == 500):
#             F2[max(0,i-25):min(w,i+25), max(0,j-25):min(h,j+25)] = 0
#
# plt.figure(figsize=(6.66,10))
# plt.imshow( (20*np.log10( 0.1 + F2)).astype(int), cmap=plt.cm.gray)
# plt.show()
# im1 = fftpack.ifft2(fftpack.ifftshift(F2)).real
# plt.figure(figsize=(10,10))
# plt.imshow(im1, cmap='gray')
# plt.axis('off')
# plt.show()
# exit()



aug_seq = iaa.Sequential([
    iaa.Resize({"height": 72*5, "width": 128*5}),
    iaa.MultiplyAndAddToBrightness(mul=(0.9,1.1), add=0),
    # 가우시안 필터는 scale 0이 정상 0~15사이인데 정상이 중간값으로 진행되지 않습니다.
    iaa.SigmoidContrast(gain=5, cutoff=0.35),
    iaa.GammaContrast((0.9,1.1), per_channel=True),
    iaa.ChangeColorTemperature(kelvin=(8000,12000)),
    iaa.MultiplyHueAndSaturation((0.8,1.1), per_channel=True),
    iaa.Resize({"height": 720, "width": 1280}),
    iaa.Sometimes(
        0.5,
        iaa.Sequential([
            iaa.MotionBlur(k=15,angle=[-90,90]),
            iaa.AdditiveGaussianNoise(scale= (0,15)),
        ])
    )

])


# dft = cv2.dft(np.float32(gray_img),flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)

# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# moire = (np.reshape(moire,(500,500,1))+1)
# print(input_img.shape)
# cv2.imshow('before',moire)
# moire = ((moire)*125).astype(np.uint8)
# cv2.imshow('after',moire)
# moire = np.resize(moire,(input_img.shape[0],input_img.shape[1]))


# moire = cv2.cvtColor(moire, cv2.COLOR_GRAY2RGB)
# input_img_t = cv2.filter2D(input_img,-1,moire)
# input_img = input_img - input_img_t + 128
# input_img = input_img + moire
print(os.listdir(img_path))
for path in os.listdir(img_path):
    img = cv2.imread(os.path.join(img_path,path))
    cv2.imshow('input_img',aug_seq.augment_image(img))
    # cv2.imshow('original',input_img)
    # cv2.imshow('noise_img',noise_img)

    cv2.waitKey(0)
