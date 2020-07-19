from flask import Flask, render_template,request
#from scipy.misc import imsave, imread, imresize

#from matplotlib.pyplot import imread

import numpy as np
import keras.models
import re
import sys 
import io
import os
import base64
#sys.path.append(os.path.abspath("./model"))
#from model.load import init
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
#from numpy import load
from numpy import expand_dims
#from matplotlib import pyplot
from keras.models import load_model
from matplotlib import pyplot as plt
from flask import send_file, jsonify
from PIL import Image
from matplotlib import cm




global model
#model= init()

model = load_model('model_017528.h5')
app = Flask(__name__)

def get_encoded_img(img):
    #img = Image.open(image_path, mode='r')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return my_encoded_img

	
# prevent cached responses
if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response

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
	pixels = load_image("output.png")
	# print('Loaded', pixels.shape)
	# generate image from source
	gen_image = model.predict(pixels)
	# print (gen_image[0].shape)
	def rescale(arr):
		arr_min = arr.min()
		arr_max = arr.max()
		return (arr - arr_min) / (arr_max - arr_min)

	myarray = np.asarray(rescale (gen_image[0]) * 255)
	myarray = myarray.astype(int)
	# print (myarray.max())
	# print (myarray.min())
	#converted = gen_image[0] * 256
	#img = Image.fromarray(np.uint8 (myarray))
	img = Image.fromarray(np.uint8 (myarray.astype(int)), 'RGB')

	#img = Image.fromarray(gen_image[0], 'RGB')
	img.save("static/generated_image3.png")

	img = get_encoded_img(img)
	# prepare the response: data
	response_data = {"image": img}
	return send_file(img, mimetype='image/png')
	#return jsonify(response_data )

	# https://stackoverflow.com/questions/56946969/how-to-send-an-image-directly-from-flask-server-to-html using send file

	# img.show()
	# np_array2 = np.asarray(Image.open("static/a.jpg"))
	# print (np_array2.shape , "this is amsterdam")
	# img = Image.fromarray(np_array2, 'RGB')
	# img.save("static/generated_image4.png")
	# plt.imshow(gen_image[0])
	# plt.axis('off')
	# plt.savefig("static/generated_image2.png")
	#return render_template("index.html", picture=img)
	#response = "0.9"
	# https://stackoverflow.com/questions/42825157/upload-an-image-and-display-it-back-as-a-response-using-flask
	#return send_file(img, mimetype='image/png')
	#return response
	#return send_file(gen_image[0], mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
