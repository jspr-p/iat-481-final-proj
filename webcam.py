import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector;

capture = cv2.VideoCapture(1) #You may need to change the number based on which webcam you are using.
detector = HandDetector(maxHands=1) 

offset = 25
imgSize= 100

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


            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImgWhite", imgWhite)
            
    if success:
        cv2.imshow("Image", img)
    cv2.waitKey(1)
        

