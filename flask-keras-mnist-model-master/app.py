from flask import Flask, render_template,request
#from scipy.misc import imsave, imread, imresize

#from matplotlib.pyplot import imread

import numpy as np
import keras.models
import re
import sys 
import os
import base64
#sys.path.append(os.path.abspath("./model"))
#from model.load import init
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
#from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from keras.models import load_model


global model
#model= init()

model = load_model('model_017528.h5')
app = Flask(__name__)


@app.route('/')
def index_view():
    return render_template('index.html')

def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	with open('output.png','wb') as output:
	    output.write(base64.b64decode(imgstr))


def load_image(filename, size=(256,256)):
    # load image with the preferred size
    pixels = load_img(filename, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # scale from [0,255] to [-1,1] and to [0 - 1]
    pixels = (pixels - 127.5) / 127.5
    pixels = (pixels + 1)/2.0
    # reshape to 1 sample
    pixels = expand_dims(pixels, 0)
    return pixels




@app.route('/predict/',methods=['GET','POST'])
def predict():
	imgData = request.get_data()
	convertImage(imgData)

	# prediction here---

	# load source image
	pixels = load_image("output.png")
	print('Loaded', pixels.shape)
	# generate image from source
	gen_image = model.predict(pixels)

	# pyplot.imshow(gen_image[0])
	# pyplot.axis('off')
	# pyplot.show()

	with open('output2.png','wb') as output:
	    output.write(base64.b64decode(gen_image))

	response = 0.9

	return response

	

if __name__ == '__main__':
    app.run(debug=True, port=8000)
