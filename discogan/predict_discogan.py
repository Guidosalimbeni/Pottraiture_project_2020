from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
import cv2

datasetpath = "./images"
outputpath = "/output"
datasetname = "/morandi"

# =============================================================================
# path = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\allGan\\Keras-GAN\\discogan\\images\\"
# name = "pix_0016.png"
# 
# def imread(path):
#     return scipy.misc.imread(path + name, mode='RGB').astype(np.float)
# 
# imgs_A = imread(path)
# 
# imgs_A = np.array(imgs_A)/127.5 - 1.
# 
# plt.imshow(imgs_A)
# 
# 
# imgs_A = imgs_A.reshape(1,256,256,3)
# =============================================================================







modelpath = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\allGan\\Keras-GAN\\saved_models\\discogan_comp\\"
from keras.models import model_from_json
# load json and create model
json_file = open(modelpath +'generator_AB.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
g_AB = model_from_json(loaded_model_json)
# load weights into new model
g_AB.load_weights(modelpath +"generator_AB_weights.hdf5")
print("Loaded model from disk")


import pyscreenshot as ImageGrab

x = 220
y = 400

while True:
    screencapture = ImageGrab.grab(bbox=(x,y,x + 256,y + 256))
    
    screencapture_mp = np.array(screencapture)
    frame = cv2.cvtColor(screencapture_mp, cv2.COLOR_BGR2GRAY)
    
    
    imgs_A = np.array(frame)/127.5 - 1.
    imgs_A = imgs_A.reshape(1,256,256,3)
    fake_B = g_AB.predict(imgs_A)
    fake_B = 0.5 * fake_B + 0.5
    cv2.imshow('frame', fake_B[0])
    

    if cv2.waitKey(1) == 13:
        break


cv2.destroyAllWindows()



# =============================================================================
# fake_B = g_AB.predict(imgs_A)
# fake_B = 0.5 * fake_B + 0.5
# plt.imshow(fake_B[0])
# 
# =============================================================================
