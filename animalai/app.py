import os
import time
import numpy as np
import cv2
from object_detection import ObjectDetectionTensorflow

colors = {
	"box" : (0, 255, 255),
	"goldenBall" : (176, 129, 25),
	"greenBall" : (148, 210, 83),
	"ramp" : (255, 0, 255),
	"redBall" : (249, 48, 17),
}

path = "/media/ivan/5a98638f-057c-4499-87fa-d0d7b41f24b4/home/ivan/Videos/animalAI/bonitos/"
detect_objects = ObjectDetectionTensorflow()
# out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30,(300,300))

for file in os.listdir(path):
	video_path = os.path.join(path, file)
	vidcap = cv2.VideoCapture(video_path)
	success = True
	while success:
		success, frame = vidcap.read()
		if not success: break
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = detect_objects.object_detection(frame)
		if results:
			for _object in results:
				_x = int(_object['topLeftX'] )
				_y = int(_object['topLeftY'])
				w = int(_object['width'])
				h = int(_object['height'])
				category = _object['category']
				cv2.rectangle(frame,(_x,_y),(_x + w,_y + h),colors[category],2)
				cv2.putText(frame, category,(_x,_y - 10),0,1,colors[category])
		cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

vidcap.release()
# out.release()
cv2.destroyAllWindows()