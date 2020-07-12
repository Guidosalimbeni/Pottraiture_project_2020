import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import cv2




from keras.models import model_from_json
# load json and create model



class App:
    def __init__(self, window, window_title, video_source=0):
         self.window = window
         self.window.title(window_title)
         self.video_source = video_source
 
         # open video source (by default this will try to open the computer webcam)
         self.vid = MyVideoCapture(self.video_source)
 
         # Create a canvas that can fit the above video source size
         self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
         self.canvas.pack()
 
         # Button that lets the user take a snapshot
         self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
         self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
 
         # After it is called once, the update method will be automatically called every delay milliseconds
         self.delay = 1
         self.modelpath = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\allGan\\Keras-GAN\\saved_models\\discogan_comp\\"
         json_file = open(self.modelpath +'generator_AB.json', 'r')  
         loaded_model_json = json_file.read()
         json_file.close()
         self.g_AB = model_from_json(loaded_model_json)
         # load weights into new model
         

         
         self.update()
 
         self.window.mainloop()
 
    def snapshot(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()
 
         if ret:
             cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
 
    def update(self):
         # Get a frame from the video source
         
         
         ret, frame = self.vid.get_frame()
         
         frame = cv2.resize(frame, (256,256))
         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
         frame = cv2.Canny(frame, 100,200)
         frame = cv2.bitwise_not(frame)
         
         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
         imgs_A = np.array(frame)/127.5 - 1.
         imgs_A = imgs_A.reshape(1,256,256,3)
         
         self.g_AB.load_weights(self.modelpath +"generator_AB_weights.hdf5")
         
         time.sleep(1)
         
         fake_B = self.g_AB.predict(imgs_A)
         fake_B = 0.5 * fake_B + 0.5  
         
         frame = cv2.resize(fake_B[0], (512,512))

         
         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         frame = np.array(frame)
         
         
         
 
         if ret:
             self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
             self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
 
         self.window.after(self.delay, self.update)
 
 
class MyVideoCapture:
     def __init__(self, video_source=0):
         # Open the video source
         self.vid = cv2.VideoCapture(video_source)
         if not self.vid.isOpened():
             raise ValueError("Unable to open video source", video_source)
 
         # Get video source width and height
         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
     def get_frame(self):
         if self.vid.isOpened():
             ret, frame = self.vid.read()
             if ret:
                 # Return a boolean success flag and the current frame converted to BGR
                 return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
             else:
                 return (ret, None)
         else:
             return (ret, None)
 
     # Release the video source when the object is destroyed
     def __del__(self):
         if self.vid.isOpened():
             self.vid.release()
 
 # Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")