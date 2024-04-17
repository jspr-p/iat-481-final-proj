import cv2
import numpy as np
import math
from ultralytics import YOLO
from cvzone.HandTrackingModule import HandDetector;
import torch as t
import threading

capture = cv2.VideoCapture(1) #You may need to change the number based on which webcam you are using.
detector = HandDetector(maxHands=1) 

offset = 25
imgSize= 200

asl_model = YOLO('runs/classify/train5/weights/best.pt')
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# target_img = np.ones((imgSize,imgSize,3),np.uint8)*255
        
def predict(im):
    prediction = asl_model(im)
    for r in prediction:
        top1index = r.probs.top1
        # print(classes[top1index])
        return classes[top1index]


if __name__ == '__main__':
    while True:
        success, img = capture.read()
        hands, img = detector.findHands(img, draw=False) #set draw to True if you want to see the hand skeleton

        if hands:
            hand = hands[0]
            x,y,w,h = hand['bbox']

            imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            
            if imgCrop.any():
                aspectRatio = h/w
                if aspectRatio > 1:
                    k = imgSize/h
                    wCal = math.ceil(k*w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    imgWhite[0:imgResizeShape[0],0:imgResizeShape[1]] = imgResize

                elif aspectRatio < 1:
                    k = imgSize/w
                    hCal = math.ceil(k*h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    imgWhite[0:imgResizeShape[0],0:imgResizeShape[1]] = imgResize
                    
                cv2.putText(img, predict(imgWhite), (x, y-20), cv2.FONT_HERSHEY_COMPLEX,2, (0,0,0), 2)
                # target_img = imgWhite
                # thread.join()
                # cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImgWhite", imgWhite)

        if success:
            cv2.imshow("Image", img)
        cv2.waitKey(1)
    