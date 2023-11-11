import cv2
import streamlit as st
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



















def main():
    st.title("Real-time Webcam Capture")
    st.write("Click 'Start' to begin capturing video from the webcam.")

    is_started = st.checkbox("Start")

    video_placeholder = st.empty()
    cap = cv2.VideoCapture(0)

    while is_started:
        ret, img = cap.read()
        if not ret:
            st.write("Error: Cannot capture video.")
            break

        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color encoding
        
        faceCurrFrame = face_recognition.face_locations(img)
        encodeCurrFrame = face_recognition.face_encodings(img)
        
        for faceLoc in faceCurrFrame:
            top, right, bottom, left = faceLoc
            encodeFace = face_recognition.face_encodings(img, [faceLoc])[0]
            
            matches = [False] * len(encodeListKnown)
            for i, known_face_encoding in enumerate(encodeListKnown):
                if known_face_encoding is not None:
                    match = face_recognition.compare_faces([known_face_encoding], encodeFace, tolerance=0.6)[0]
                    matches[i] = match

            if any(matches):
                matchIndex = matches.index(True)
                name = class_names[matchIndex].upper()
                
                cvzone.cornerRect(imgS, (left, top), (right, bottom), (0, 255, 0), 2)
                cvzone.cornerRect(imgS, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cvzone.putTextRect(imgS, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                markAttendance(name)
        
        video_placeholder.image(imgS, channels="RGB")  # Specify color channels for proper display
    
    cap.release()

if __name__ == "__main__":
    main()
#In this code, the width parameter in the st.image method is set to 640. You can adjust this value to change the displayed width of the webcam feed in Streamlit.






