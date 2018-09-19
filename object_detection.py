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
        self.model_name = 'ssd_mobilenet_v1_coco_2018_01_28'
        self.path_to_ckpt = self.model_name + '/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph)

    def create_response(self, image_np, boxes, classes, scores):
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
        iterator = 0
        height, width, layers = image_np.shape
        while scores[iterator] > 0.5 and iterator < len(scores):
            (y_min, x_min, y_max, x_max) = (boxes[iterator, 0], boxes[iterator, 1], \
                                            boxes[iterator, 2], boxes[iterator, 3])
            category = CATEGORY_INDEX[str(int(classes[iterator]))]['name']
            confidence = scores[iterator]
            top_left_x = round(x_min * width)
            top_left_y = round(y_min * height)
            _height = round(height * (y_max - y_min))
            _width = round(width * (x_max - x_min))

            detected_objects.append(dict(category=category, confidence=float(confidence), \
                                         topLeftX=top_left_x, topLeftY=top_left_y, height=_height, width=_width))

            iterator += 1
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
            # Definite input and output Tensors for self.detection_graph
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            image_np_expanded = np.expand_dims(image, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = self.sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            response = self.create_response(image, np.squeeze(boxes), np.squeeze(classes),
                                            np.squeeze(scores))
            if response:
                return response
            return None
        except Exception as error:
            return dict(OurFault="Can not make the object detection processing " + str(error))
