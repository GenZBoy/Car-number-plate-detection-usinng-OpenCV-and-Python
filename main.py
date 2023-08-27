# -*- coding: utf-8 -*-
"""
Created on Sun Aug  27 20:36:33 2023

@author: soumyo
"""

'''processing a video'''
import cv2
import pytesseract
import time
import imutils
harcascade = "haarcascade_russian_plate_number.xml"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

car_cascade = cv2.CascadeClassifier('cars.xml')

def perform_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)

video_path = 'rashmalai.mp4'
cap = cv2.VideoCapture(video_path)

cap.set(3, 640) # width
cap.set(4, 480) #height

min_area = 500
count = 0

time_gap = 2  

while cap.isOpened():
    ret, frame = cap.read()
   
    
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = imutils.resize(frame, width=600)
    
    cars = car_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    plate_cascade = cv2.CascadeClassifier(harcascade)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    #previous file 
    # for (x, y, w, h) in cars:
    #     car_image = frame[y:y+h, x:x+w]
    #     number_plate_text = perform_ocr(car_image)
        
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #     cv2.putText(frame, number_plate_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    #     cv2.imshow('Number Plate Detection', frame)

    for (x,y,w,h) in plates:
        area = w * h
        car_image = frame[y:y+h, x:x+w]
        number_plate_text = perform_ocr(car_image)
        if area > min_area:
         
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, number_plate_text, (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            print(number_plate_text)
            img_roi = frame[y: y+h, x:x+w]
            cv2.imshow("Number plate detection", img_roi)


    
    cv2.imshow("Result", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("plates/scaned_img_" + str(count) +number_plate_text+".jpg", img_roi)
        cv2.rectangle(frame, (0,200), (640,300), (0,255,0), cv2.FILLED)
        cv2.putText(frame, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results",frame)
        cv2.waitKey(500) 
        count += 1
    

   


cap.release()
cv2.destroyAllWindows()
