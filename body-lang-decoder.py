import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
# import csv
import threading
import sys
import socket 
import os
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
num_fml = 0
extemp_connected = False
left_wrist = 0.0

server_address = (sys.argv[1], int(sys.argv[2]))
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cap = cv2.VideoCapture(0)
threadStopped = False

def calcParam(connection):
    # at regular intervals, calculate summaries and send to extempore
    time.sleep(0.25)
    connection.send(str(left_wrist).encode())
    return

try:
    server_sock.bind(server_address)
except socket.error as e: 
    print(str(e))
print("Extempore socket bound")
server_sock.listen(1)


while not extemp_connected: 
    connection, address = server_sock.accept()
    p_thread = threading.Thread(target=calcParam, args=(connection, ))
    p_thread.start()
    extemp_connected = True

# Initiate holistic model
    # print("res: "+str(cap.get(cv2.CAP_PROP_FRAME_WIDTH))+"X"+str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    flag = True
    while cap.isOpened():
        ret, frame = cap.read()
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)
        if(results.left_hand_landmarks):
            left_wrist = results.left_hand_landmarks.landmark[0].x
        # face_landmarks (468), pose_landmarks (33), left_hand_landmarks (21), right_hand_landmarks (21)
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # # Export coordinates
        # try:
        #     # Extract Pose landmarks
        #     # pose = results.pose_landmarks.landmark
        #     # pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
        #     # Extract Face landmarks
        #     face = results.face_landmarks.landmark
        #     face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            
        #     # # Concate rows
        #     # row = face_row
            
        #     # # Append class name 
        #     face_row.insert(0, "test")
            
        #     # Export to CSV
        #     with open('coords.csv', mode='a', newline='') as f:
        #         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #         csv_writer.writerow(face_row) 
            
        # except:
        #     print("failed to export")
        #     pass
        # 1. Draw face landmarks
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                                #  mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                #  mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                #  )
        
        # 2. Right hand
        # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        #                          mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        #                          mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        #                          )

        # # 3. Left Hand
        # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        #                          mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        #                          mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        #                          )

        # # 4. Pose Detections
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
        #                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        #                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        #                          )
                        
        # cv2.imshow('Video Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            # threadStopped = True
            # p_thread.join()
            break

'''
Tasks
- read web feed - handle keyboard interaction. Like sensors
- server's main function to make predictions, bind to socket
- at regular time intervals, send data to extempore
'''