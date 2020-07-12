#import tensorflow.lite as lite
from tensorflow import lite
import tensorflow as tf
from keras.models import load_model
print (tf.__version__)

# keras_model_path = "model_017528.h5"

# tflite_model_path = "a.txt"


# keras_model= load_model(keras_model_path, compile=False)

# converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)



# tflite_model = converter.convert()


# file = open( tflite_model_path , 'wb' ) 
# file.write( tflite_model)


keras_model_path = "model_017528.h5"

tflite_model_path = "model_017528.tflite"

#model= load_model(keras_model_path) # , compile=False
model=tf.keras.models.load_model(keras_model_path, compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
tflite_model = converter.convert()

# converter = tf.lite.TFLiteConverter.from_keras_model(keras_model_path)

# tflite_model = converter.convert()

file = open( tflite_model_path , 'wb' ) 
file.write( tflite_model)

# open(tflite_model_path, "wb").write(tflite_model)
# converter = lite.TFLiteConverter.from_keras_model(keras_model_path) # https://colab.research.google.com/drive/1IUIn9ffk5ICKujqPyuGaHL2irQ9Wmtpm

# https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/codelabs/digit_classifier/ml/step2_train_ml_model.ipynb#scrollTo=2fXStjR4mzkR
# for quant and prediction of tflite model