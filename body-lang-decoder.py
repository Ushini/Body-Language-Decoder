import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import threading
import sys
import socket 
import os
import time
import numpy as np
from pythonosc import udp_client

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
num_fml = 0
extemp_connected = False
server_address = (sys.argv[1], int(sys.argv[2]))
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cap = cv2.VideoCapture(0)

def calcParam(connection):
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        flag = True
        # print("res: "+str(cap.get(cv2.CAP_PROP_FRAME_WIDTH))+"X"+str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(image)
            
            # face_landmarks (468), pose_landmarks (33), left_hand_landmarks (21), right_hand_landmarks (21)
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            
            # cv2.imshow('Video Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

try:
    server_sock.bind(server_address)
except socket.error as e: 
    print(str(e))
print("Extmp socket bound")
server_sock.listen(1)

while not extemp_connected: 
    connection, address = server_sock.accept()
    p_thread = threading.Thread(target=calcParam, args=(connection, ))
    p_thread.start()
    extemp_connected = True