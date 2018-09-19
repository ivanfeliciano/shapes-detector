import numpy as np
import cv2
from object_detection import ObjectDetectionTensorflow

detect_objects = ObjectDetectionTensorflow()
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    results = detect_objects.object_detection(frame)
    if results:
        for _object in results:
            print(_object)
            _x = int(_object['topLeftX'] )
            _y = int(_object['topLeftY'])
            w = int(_object['width'])
            h = int(_object['height'])
            category = _object['category']
            cv2.rectangle(frame,(_x,_y),(_x + w,_y + h),(0,255,0),2)
            cv2.putText(frame, category,(_x,_y - 10),0,1,(0,255,0))
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()