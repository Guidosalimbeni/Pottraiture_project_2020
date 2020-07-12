import cv2
import numpy as np
    
cap = cv2.VideoCapture(0)

edged = True
inverted = True
background = cv2.imread('base_background.jpg')

count = 0
stop = False
captureSent =  np.zeros((365,365,3), np.uint8)
#capturePrediction = np.zeros((365,365,3), np.uint8)


from threading import Thread

def runA ():
    while (stop == False):
        
        global capturePrediction
        capturePrediction = captureSent.copy()
        capturePrediction = cv2.flip(capturePrediction, 1)
        


def runB ():
    while (stop == False):
        pass
        
        

qt1 = Thread(target = runA)
t2 = Thread (target = runB)
t1.setDaemon(True)
t2.setDaemon(True)
t1.start()
t2.start()




while True:

    ret, capture = cap.read()
    
    if not ret:
        break

    
    capture = capture[112:112+256, 192:192+256]
    
    
    background = cv2.resize(background, (1920,1080))
    
    xA_offset= 590 + 365 + 11
    yA_offset= 358
    capture = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    capture = cv2.Canny(capture, 10,255)
    capture = cv2.resize(capture, (365,365))
    capture = cv2.cvtColor(capture, cv2.COLOR_GRAY2BGR)
    capture = cv2.bitwise_not(capture)
    capture = cv2.flip(capture, 1)
    
    captureSent = capture.copy()
    
    background[yA_offset:yA_offset+capture.shape[0], xA_offset:xA_offset+capture.shape[1]] = capture
    try:
        cv2.imshow(' Salimbeni', capturePrediction)
        cv2.waitKey(1)
        print (capturePrediction.shape)
    except:
        print ("opssss")
    
    cv2.imshow('Pot-traiture by Guido Salimbeni', background)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop = True
        break
    
    count += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




