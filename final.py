# python3 final.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import tkinter.messagebox
from tkinter import *
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import config
import smtplib
import os
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
vs = VideoStream(src=0).start()
time.sleep(2.0)
rt = Tk()
rt.withdraw()
thresh = 02.00
global flag
flag = True


def SendMail():
    msg = MIMEMultipart()
    msg['Subject'] = 'Subject'
    msg['From'] = config.DATA_Username
    msg['To'] = 'ssonje97@gmail.com'
    text = MIMEText('Warning!')
    msg.attach(text)
    s = smtplib.SMTP('smtp.gmail.com',587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login(config.DATA_Username,config.DATA_Password)
    s.sendmail(config.DATA_Username,'ssonje97@gmail.com',msg.as_string())
    s.quit()

def ImgMail(ImgFileName):
    img_data = open(ImgFileName,'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = 'Subject'
    msg['From'] = config.DATA_Username
    msg['To'] = 'ssonje97@gmail.com'
    image = MIMEImage(img_data,name=os.path.basename(ImgFileName), _subtype="jpeg")
    msg.attach(image)
    s = smtplib.SMTP('smtp.gmail.com',587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login(config.DATA_Username,config.DATA_Password)
    s.sendmail(config.DATA_Username,'ssonje97@gmail.com',msg.as_string())
    s.quit()


while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    orig = frame.copy()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            if (CLASSES[idx]  == 'bottle'):
                                if flag:
                                        flag = False
                                        SendMail()
                                choice = tkinter.messagebox.askquestion("Danger","Do you want to send Email?")
                                if choice == 'yes':
                                        crop_image = orig
                                        cv2.imwrite('Cropped.png',crop_image)
                                        ImgMail('Cropped.png')
                                        tkinter.messagebox.showinfo('Notification','This can be danger')
                                        cv2.destroyAllWindows()
                                        vs.stop()
                                        break
                                elif choice == 'no':
                                        thresh = thresh+10
                                        continue
                                        
            

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    



cv2.destroyAllWindows()
vs.stop()
