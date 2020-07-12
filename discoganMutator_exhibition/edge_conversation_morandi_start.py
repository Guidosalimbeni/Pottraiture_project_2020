import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import os
from skimage import feature, io
from scipy import misc
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from skimage.color import rgba2rgb
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io, color, img_as_float
from skimage import util 
from keras.preprocessing.image import img_to_array
from numpy import asarray
from numpy import savez_compressed
from skimage.transform import resize
# https://medium.com/@yanweiliu/how-to-use-tensorflow-lite-in-flutter-e323422f64b8
# https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/

# https://www.digitalknights.co/blog/build-computer-vision-ios-app-in-flutter

pathA = "D:\\google drive\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\Anlysis of Comp ML compml 2020\\CNN_fromUnity\\good"

#outputPathA_B ="D:\\mutatorDiscoGan\\"
outputPathA_B ="D:\\morandi_edges"

def imagelist (path):
    imagepaths = []
    imageNames = []
    valid_images = [".jpg", ".png", ".tga", ".gif"]
    for f in os.listdir(path):
    
        ext = os.path.splitext(f)[1]
        name = os.path.splitext(f)[0]
        if ext.lower() not in valid_images:
            continue
        imagepaths.append(os.path.join(path,f))
        imageNames.append(name)
    # store the total number of images
    totalNumOfImages = len(imagepaths) 
    
    return imagepaths, imageNames, totalNumOfImages

image_in = io.imread("D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\Anlysis of Comp ML compml 2020\\CNN_fromUnity\\good\\1_1_savedimage.png"
)

path = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\Anlysis of Comp ML compml 2020\\CNN_fromUnity\\good\\"

#print (image_rescaled.shape) # 180, 240 to 256,256

def load_images(path):
    src_list, tar_list = list(), list()
    for image_file in os.listdir(path):
        imageraw_ = io.imread(path + image_file)
        im = rgb2gray(imageraw_)
        image = feature.canny(im, sigma=1) # ok with sigma 1
        image = util.invert(image)
        image_rescaled = rescale(image, 1.43, anti_aliasing=False)
        image_rescaled = image_rescaled[:, 44:]
        image_rescaled = image_rescaled[:, :-43]
        image_rescaled = image_rescaled[1:, :]
        image_rescaled = gray2rgb(image_rescaled)


        image_in = rgba2rgb(imageraw_)
        img_resized = resize(image_in, (256, 343))
        im_rescaled = img_resized[:, 44:]
        im_rescaled = im_rescaled[:, :-43]

        pixels_edges = img_to_array(image_rescaled)
        images = img_to_array(im_rescaled)
        
        src_list.append(images)
        tar_list.append(pixels_edges)

    return [asarray(src_list), asarray(tar_list)]
#load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'maps_256.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)
