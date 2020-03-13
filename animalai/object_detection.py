# coding: utf-8
"""
.. module:: object_detection
   :platform: Unix, Windows
   :synopsis: Módulo para la detección de objetos en una fotografia usando la API de TensorFlow models.

.. moduleauthor:: Ivan Feliciano <ivan.felavel@gmail.com>

"""
import os
import numpy as np
import tensorflow as tf
import json
import base64


CATEGORY_INDEX = {}
try:
    with open('category_idx.json', 'r') as fp:
        CATEGORY_INDEX = json.load(fp)
except FileNotFoundError as error:
    print("You need the categories JSON file {}".format(error))
    raise error


class ObjectDetectionTensorflow(object):

    def __init__(self):
        """El constructor se encarga de cargar el modelo de Tensorflow en la
        memoria, a partir del archivo ``frozen_inference_graph.pb``.
        """
        self.model_name = 'inference_graph/saved_model'
        self.model = tf.saved_model.load(self.model_name)
        self.model = self.model.signatures['serving_default']

    def create_response(self, image_np, boxes, classes, scores, num_detections):
        """.. :method:: create_response(image_np, classes, scores)

        Método que se encarga de convertir en un diccionario el resultado del
        modelo, para que pueda ser regresado en el cuerpo de la API REST.

        El formato del objeto retornado es el siguiente:

        .. code-block:: javascript

            [
                {
                    "confidence" : 0.0,
                    "topLeftX": 0.0,
                    "topLeftY": 0.0,
                    "height" : 0.0
                    "width" : 0.0
                    "category" : ""
                }
            ]


        :param image_np: El contenedor de la imagen.
        :type image_np: numpy.array.
        :param boxes: Un arreglo con las coordenadas del cuadrilátero que encierran a cada objeto.
        :type boxes: numpy.array
        :param scores: Una lista con las probabilidades de que el objeto encontrado sea el correcto.
        :type scores: list
        :return: Un diccionario ``response`` que contiene los objectos detectados.
        :rtype: dict



        """
        response = None
        detected_objects = []
        box_idx = 0
        height, width, layers = image_np.shape
        boxes = boxes[0]
        for i in range(num_detections):
            if scores[i] < 0.8:
                continue
            y_min, x_min, y_max, x_max = boxes[i]
            category = CATEGORY_INDEX[str(int(classes[i]))]['name']
            confidence = scores[i]
            top_left_x = round(x_min * width)
            top_left_y = round(y_min * height)
            _height = round(height * (y_max - y_min))
            _width = round(width * (x_max - x_min))
            detected_objects.append(dict(category=category, confidence=float(confidence), \
                                         topLeftX=top_left_x, topLeftY=top_left_y, height=_height, width=_width))
        if len(detected_objects) > 0:
            response = detected_objects
        return response
        

    def object_detection(self, image, is_url=None):
        """.. :method:: object_detection(image, is_url):

        Es el método encargado de evaluar la imagen en el modelo. Aquí es donde
        se realiza la ejecución de la gráfica de tensorflow.
        Primero corre el modelo, después genera la repuesta con :meth:`create_response`
        y regresa el resultado de éste, dependiendo de si encontró o no objectos.

        :param image: La imagen que se desea procesar.
        :type image: string.
        :param is_url: Bandera para elegir la forma en que se carga la imagen.
        :type is_url: boolean.
        :return: La respuesta que enviará la API REST.
        :rtype: dict.

        """
        try:
            image = np.asarray(image)
            input_tensor = tf.convert_to_tensor(image)
            input_tensor = input_tensor[tf.newaxis,...]
            output_dict = self.model(input_tensor)
            num_detections = int(output_dict.pop('num_detections'))
            output_dict = {key:value[0, :num_detections].numpy()\
                            for key,value in output_dict.items()}
            output_dict['num_detections'] = num_detections
            boxes = output_dict['detection_boxes'],
            scores = output_dict['detection_scores']
            classes = output_dict['detection_classes']
            num = output_dict['num_detections']
            response = self.create_response(image, boxes, classes, scores, num_detections)
            return response
            
        except Exception as error:
            print(error)
            return dict(OurFault="Can not make the object detection processing " + str(error))
