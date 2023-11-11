# IMPORTING
import cv2
import pandas as pd
#import cvzone
import numpy as np
import face_recognition
import os
import cvzone
from datetime import datetime

def markAttendance(name):
    with open("attendance.csv" ,"r+") as f:
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry =line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dString}')


# COLLECTION OF IMAGES
path = "images_attendance"
images = []
class_names = []
myList = os.listdir(path)
print(myList)
for cls in myList:
    currImg = cv2.imread(f'{path}/{cls}')
    images.append(currImg)
    class_names.append(os.path.splitext(cls)[0])   # remove .jpeg extension
print(class_names)   #name 
print(images)   #array


def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img)
        
        if len(face_locations) > 0:
            encode = face_recognition.face_encodings(img, [face_locations[0]])[0]  # Get the first face encoding
            encodeList.append(encode)
        else:
            encodeList.append(None)
    return encodeList

encodeListKnown = find_encodings(images)
# WEBCAM AND MATCHING IMAGES
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    
    
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    faceCurrFrame = face_recognition.face_locations(img)
    encodeCurrFrame = face_recognition.face_encodings(img)
    
    for faceLoc in faceCurrFrame:
        top, right, bottom, left = faceLoc
        encodeFace = face_recognition.face_encodings(img, [faceLoc])[0]  # Get the first face encoding
        
        matches = [False] * len(encodeListKnown)
        for i, known_face_encoding in enumerate(encodeListKnown):
            if known_face_encoding is not None:
                match = face_recognition.compare_faces([known_face_encoding], encodeFace, tolerance=0.6)[0]
                matches[i] = match

        if any(matches):
            matchIndex = matches.index(True)
            name = class_names[matchIndex].upper()
            
            
            # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            # cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            # cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cvzone.cornerRect(img, (left, top, right - left, bottom - top),  l=10 , t=3)


            # Add text
            cvzone.putTextRect(img, f"PERSON : {name}", (50,50))

            markAttendance(name)

    cv2.imshow("webcam", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
