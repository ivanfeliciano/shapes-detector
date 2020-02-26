import numpy as np
import cv2
from object_detection import ObjectDetectionTensorflow
import yagmail


# pip install keyring
# pip install yagmail

detect_objects = ObjectDetectionTensorflow()
cap = cv2.VideoCapture(0)
done = False
img_filename = 'img.png'
while not done:
	ret, frame = cap.read()
	results = detect_objects.object_detection(frame)
	if results:
		for _object in results:
			_x = int(_object['topLeftX'] )
			_y = int(_object['topLeftY'])
			w = int(_object['width'])
			h = int(_object['height'])
			category = _object['category']
			cv2.rectangle(frame,(_x,_y),(_x + w,_y + h),(0,255,0),2)
			cv2.putText(frame, category,(_x,_y - 10),0,1,(0,255,0))
			if category == 'person':
				print("find person")
				cv2.imwrite(img_filename, frame)
				contents = [
					"Esto es un mensaje de prueba envidado desde un correo de Office365.",
					"Puedes encontrar la imagen adjunta.", img_filename
				]
				# yagmail.SMTP('deserted.alberte').send('vglr3000@gmail.com', 'subject', contents)
				yag = yagmail.SMTP('sr.raton@comunidad.unam.mx', 'Abrete19.94', host='smtp.office365.com', port=587, smtp_starttls=True, smtp_ssl=False)
				yag.send(['ivan.felavel@gmail.com', 'ivan_meridbike@gmail.com'], 'subject', contents)
				done = True
				break
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()