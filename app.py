import numpy as np
import cv2
from shape_detector import ShapeDetector
from object_detection import ObjectDetectionTensorflow

detect_objects = ObjectDetectionTensorflow()
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    results = detect_objects.object_detection(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret,thresh = cv2.threshold(blurred,127,255,1)
    img, cnts, h = cv2.findContours(thresh,1,2)
    # sd = ShapeDetector()

    # for c in cnts:
    #     shape = sd.detect(c)
    #     M = cv2.moments(c)
    #     if M['m00'] > 0:
    #         cX = int((M["m10"] / M["m00"]))
    #         cY = int((M["m01"] / M["m00"]))
    #         print(dict(shape=shape, x=cX, y=cY))
    #         cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
    #         cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
    #         0.5, (255, 255, 255), 2)
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