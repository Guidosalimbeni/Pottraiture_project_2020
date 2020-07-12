from __future__ import print_function, division
#import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
#import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
#import os
import cv2

datasetpath = "./images"
outputpath = "/output"
datasetname = "/mutator"

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

# this is for the morandi weights... CONTOURS and faces but selected only 200 images and edited a bit in photoshop
#modelpath = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\allGan\\Keras-GAN\\saved_models\\morandiselection\\"

# this is for the edge COMBINED edge contours version and composition morandi 400 images
#modelpath = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\allGan\Keras-GAN\\saved_models\\morandicombined\\"

# this is for the edge celeb version and composition morandi
#modelpath = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\allGan\Keras-GAN\\saved_models\\celebedgedmorandi\\"


# this is for the morandi weights... CONTOURS - exhibition  !!!!!!!!!!
#modelpath = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\allGan\\Keras-GAN\\saved_models\\discogan_comp\\"
modelpath = ".\\saved_models\\discogan_comp\\"
# this is for the new celeb morandi weights...(photo in color from celeb dataset ... circa 145 epochs on 200 photos..) NOT LEARNED MUCH
#modelpath = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\allGan\\Keras-GAN\\saved_models\\celebMorandi\\"

# this is for mutator
#modelpath = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\allGan\\Keras-GAN\\discoganMutator\\saved_model\\"


# this is for the edge celeb version and composition morandi BIGGER database -- NOT LEARNED ANYTHING TOO HIGH LOSS
#modelpath = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\allGan\Keras-GAN\\saved_models\\celebBigMorandi\\"


from keras.models import model_from_json
# load json and create model
json_file = open(modelpath +'generator_AB.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
g_AB = model_from_json(loaded_model_json)
# load weights into new model
g_AB.load_weights(modelpath +"generator_AB_weights.hdf5")
print("Loaded model from disk")


application = 'webcam'

#application = 'grabscreen'

# this was working with the MORANDI DATASET .. didn't try with mutator
if application == 'grabscreen':

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


edged = False
inverted = True
### this bit if for the webcam
if application == 'webcam':
    
    cap = cv2.VideoCapture(0)
    
    background = cv2.imread('base_background.jpg')
    background = cv2.resize(background, (256,256))
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    while True:
        # Capture frame-by-frame
        ret, capture = cap.read()
        
        if not ret:
            break
        
        #webcapture = np.array(capture)
        # Our operations on the frame come here
        
        #cv2.imshow('frame1', capture)

        capture = capture[112:112+256, 192:192+256]
        
        frame = cv2.resize(capture, (256,256))
        
        if edged:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #frame = cv2.Canny(frame, 100,255)
            frame = cv2.Canny(frame, 100,255)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if inverted:
                frame = cv2.bitwise_not(frame)
        
        imgs_A = np.array(frame)/127.5 - 1.
        imgs_A = imgs_A.reshape(1,256,256,3)
        fake_B = g_AB.predict(imgs_A)
        fake_B = 0.5 * fake_B + 0.5
        
        bigger = cv2.resize(fake_B[0], (256,256))
        
# =============================================================================
#         height = 720
#         width = 1280
#         background = np.zeros((height,width,3), np.uint8)
#         background[:,:] = (255,0,0)      # (B, G, R)
#         
#         
#         x_offset=384
#         y_offset=232
#         #background = cv2.imread('base_background.jpg')
#         background[y_offset:y_offset+bigger.shape[0], x_offset:x_offset+bigger.shape[1]] = bigger
# =============================================================================
        
        #forcapture = cv2.resize(capture, (256,256))
        bigger = cv2.cvtColor(bigger, cv2.COLOR_BGR2GRAY)
        
        #background = np.zeros((256,256) , np.uint8)
        #background = cv2.resize(background, (256,256))
        #forcapture = cv2.cvtColor(forcapture, cv2.COLOR_RGB2GRAY)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        vis = np.concatenate((bigger, frame), axis=1)
        
        cv2.imshow('frame1', vis)
        #cv2.imshow('cap', capture)
        #cv2.imshow('frame2', fake_B[0])
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



# =============================================================================
# fake_B = g_AB.predict(imgs_A)
# fake_B = 0.5 * fake_B + 0.5
# plt.imshow(fake_B[0])
# 
# =============================================================================
