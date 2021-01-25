# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import cv2

from tkinter import *
import cv2
import numpy as np
from PIL import Image, ImageTk
import face_recognition
import os
import PIL
import shutil
from autocrop import Cropper
from datetime import datetime
import pickle
# from sklearn import svm


def addBox(arrayImageHistory, dateCHeck):
    # I use len(all_entries) to get number of next free column
    i = 0
    for imageArray in arrayImageHistory:
        i += 1
        box_row = i * 2 + 1
        Button(window, image=imageArray).grid(row=box_row, column=1, ipadx=50)

    j = 0
    for timeCheck in dateCHeck:
        j += 1
        text_row = j * 2 + 2
        Label(window, text=timeCheck).grid(row=text_row, column=1, pady=20)


root = Tk()
cropper = Cropper()
root.geometry("1700x1200")
lmain = Label(root)
lmain.pack(side="right", anchor=NW)
timeSleep = 0
timeUnknow = 0
time_not_detect = 0

path = 'ImagesCompany'
images = []
arrayButton = []
classNames = []

face_locations = []
face_names = []
face_not_detect = []

imagesHistory = []
dateArrayHistory = []
process_this_frame = True

myList = os.listdir(path)
print(myList)

frameAddBox = Frame(root, relief=GROOVE, width=550, height=50, bd=1, bg="white")
frameAddBox.place(x=10, y=30)

canvas = Canvas(frameAddBox, bg="white")

fileModelSVM = "/Users/v-miodohien/Desktop/SVMFaceDetect/finalized_model.sav"
fileClassSVM = "/Users/v-miodohien/Desktop/SVMFaceDetect/finalized_ClassNames.sav"

# load the model from disk
svmModel = pickle.loads(open(fileModelSVM, 'rb').read())
svmClasses = pickle.loads(open(fileClassSVM, 'rb').read())


def myfunction(event):
    canvas.configure(scrollregion=canvas.bbox("all"), width=350, height=1150)


window = Frame(canvas, bg="white")
scrollbarBox = Scrollbar(frameAddBox, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbarBox.set)
scrollbarBox.pack(side="right", fill="y")

Label(root, text="History FaceDetect").pack(side=TOP)

canvas.pack(side="left")
canvas.create_window((0, 0), window=window, anchor='nw')
window.bind("<Configure>", myfunction)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(imagesEndcoding):
    encodeList = []
    for imgArray in imagesEndcoding:
        imgCVT = cv2.cvtColor(imgArray, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(imgCVT)[0]
        encodeList.append(encode)
    return encodeList


def clearFrame():
    for widget in window.winfo_children():
        widget.destroy()


def clearFounder():
    for rootFile, dirs, files in os.walk('HistoryFaceDetect'):
        for f in files:
            os.unlink(os.path.join(rootFile, f))
        for d in dirs:
            shutil.rmtree(os.path.join(rootFile, d))

    for rootFile, dirs, files in os.walk('FaceCropImage'):
        for f in files:
            os.unlink(os.path.join(rootFile, f))
        for d in dirs:
            shutil.rmtree(os.path.join(rootFile, d))


def readFileImage(dateCHeck):
    pathHisotry = 'HistoryFaceDetect'
    listHistory = os.listdir(pathHisotry)
    for clArray in listHistory:
        nameFile = f'{pathHisotry}/{clArray}'
        imageOpen = Image.open(nameFile)
        imageOpen = imageOpen.resize((250, 250), Image.ANTIALIAS)
        addImageArray = ImageTk.PhotoImage(imageOpen)
        imagesHistory.insert(0, addImageArray)

    if len(imagesHistory) != 0:
        addBox(imagesHistory, dateCHeck)


# with open('dataset_faces.dat', 'rb') as f:
#     all_face_encodings = pickle.load(f)
#
# # Create arrays of known face encodings and their names
# encodeListKnown = np.array(list(all_face_encodings.values()))

# Get a reference to webcam #0 (the default one)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Grab a single frame of video
    success, frame = cap.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    imgS = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = imgS[:, :, ::-1]
    faces = classifier.detectMultiScale(imgS)

    # Only process every other frame of video to save time
    if process_this_frame:
        face_names = []
        face_not_detect = []
        face_locations = face_recognition.face_locations(rgb_small_frame)
        encodesCurFrame = face_recognition.face_encodings(rgb_small_frame, face_locations)

        no = len(face_locations)

        # Predict all the faces in the test image using the trained classifier
        # print("Found:")
        for i in range(no):
            test_image_enc = encodesCurFrame[i]
            name = svmModel.predict([test_image_enc])
            predict_proba = svmModel.predict_proba([test_image_enc])[0]
            j = np.argmax(predict_proba)
            if predict_proba[j] > 0.6:
                timeSleep += 1
                if timeSleep % 15 == 0:
                    dateDetect = f'{name[0]}_{str(datetime.now())}'
                    dateArrayHistory.insert(0, dateDetect)
                    clearFounder()
                    clearFrame()
                    for (top, right, bottom, left) in face_locations:
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        imCrop = frame[top:bottom, left:right]
                        cv2.imwrite('HistoryFaceDetect/' + name[0] + '.jpg', imCrop)

                    readFileImage(dateArrayHistory)
                    today = datetime.today()
                    d1 = today.strftime("%d-%m-%Y")
                    file1 = open(f"DataHistoryDetect/Data_Detect_{d1}.txt", "a")
                    file1.write(f'{str(dateDetect)} \n')
                    file1.close()
                    timeSleep = 0
                else:
                    face_names.append(name)
            else:
                timeUnknow += 1
                if timeUnknow % 10 == 0:
                    dateDetect = f'Hello_Person_{str(datetime.now())}'
                    dateArrayHistory.insert(0, dateDetect)

                    clearFounder()
                    clearFrame()

                    for (top, right, bottom, left) in face_locations:
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        imCrop = frame[top:bottom, left:right]
                        cv2.imwrite('HistoryFaceDetect/' + name[0] + '.jpg', imCrop)

                        readFileImage(dateArrayHistory)
                        today = datetime.today()
                        d1 = today.strftime("%d-%m-%Y")
                        file1 = open(f"DataHistoryDetect/Data_Detect_{d1}.txt", "a")
                        file1.write(f'{str(dateDetect)} \n')
                        file1.close()
                        timeUnknow = 0
                else:
                    nameUnknown = "Unknown"
                    face_names.append(nameUnknown)
    process_this_frame = not process_this_frame

    # Display the results
    if len(face_names) != 0:
        for (top, right, bottom, left), nameBox in zip(face_locations, face_names):

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            time_not_detect = 0
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            if nameBox == "Unknown":
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, nameBox, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            else:
                # cv2.putText(frame, name1[:-2], (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                # print(nameBox[0])
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                print(f'{left},{top}, {right}, {bottom}')
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, nameBox[0], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    else:
        time_not_detect += 1
        if time_not_detect % 10 == 0:
            cv2.rectangle(frame, (494, 214), (870, 594), (0, 255, 0), 2)
            cv2.rectangle(frame, (504, 584 - 35), (860, 584), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, "Put your face", (550 + 6, 214 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                       (255, 255, 255), 1)
            cv2.putText(frame, "in this green box", (535 + 6, 580 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            time_not_detect = 9

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    root.update()
