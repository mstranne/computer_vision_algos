import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

input_file = 'input_imgs/thresholding_example.jpg'
img = cv.imread(input_file)


cv.imshow('input img', img)
cv.waitKey(0)

ret, thresh1 = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
cv.imshow('global thresh', thresh1)
cv.waitKey(0)

#global threshholding

# first lets stat with the first approach, Bernsens algo
# https://www.scirp.org/(S(351jmbntv-nsjt1aadkposzje))/reference/referencespapers.aspx?referenceid=83625
# https://ww3.ticaret.edu.tr/mckasapbasi/files/2015/09/Implementation-of-Bernsen%E2%80%99s-Locally-Adaptive-Binarization-Method.pdf

grey_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img_h, img_w, _ = img.shape

# this approach will calculate a threshhold value for small windows,
# so its a sliding window approach

win_size = 18
bin_img = np.zeros((img_h, img_w))

# todo implement with numpy sliding window
for x in range(0, img_h):
    for y in range(0, img_w):
        # thresh val is (Zlow + Zhigh) / 2, where Z is lowest and highest grey vals in window
        x_l = 0 if x < win_size else x-win_size
        x_h = img_h-1 if x > img_h-win_size else x+win_size
        y_l = 0 if y < win_size else y-win_size
        y_h = img_w-1 if y > img_w - win_size else y+win_size

        #print('{0}:{1},{2}:{3}\n'.format(x_l,x_h,y_l,y_h))
        Z = (grey_img[x_l:x_h, y_l:y_h].min()+grey_img[x_l:x_h, y_l:y_h].max() ) /2

        if grey_img[x,y] > Z :
            bin_img[x,y] = 255

cv.imshow('bernsen thresh', bin_img)
cv.waitKey(0)

# Paper 2
# https://people.scs.carleton.ca/~roth/iit-publications-iti/docs/gerh-50002.pdf


print('heilo')