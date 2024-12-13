# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:21:21 2024

@author: jperezr
"""

import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import streamlit as st

# Cargar el modelo desde TensorFlow Hub
model_url = "https://tfhub.dev/tensorflow/efficientdet/d0/1"  # Ejemplo de modelo de detección de objetos
model = hub.load(model_url)

# Función para preprocesar la imagen
def preprocess_image(image):
    # Redimensiona la imagen a 512x512 si es necesario
    image_resized = cv2.resize(image, (512, 512))
    
    # Convierte la imagen de float32 a uint8
    image_uint8 = np.uint8(image_resized)
    
    # Normaliza los valores de píxeles entre 0 y 255
    image_normalized = np.clip(image_uint8, 0, 255)
    
    # Agrega una dimensión para el lote (batch)
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch

# Función para detectar objetos en la imagen
def detect_objects(image):
    # Preprocesa la imagen
    input_tensor = preprocess_image(image)
    
    # Realiza la predicción con el modelo
    detections = model(input_tensor)
    
    return detections

# Cargar y procesar el video
cap = cv2.VideoCapture('america_monterrey.mp4')

# Crear un contenedor de imágenes en Streamlit
frame_container = st.empty()

# Inicializar un botón para detener el video
stop_button = st.button('Stop')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Realiza la detección en el cuadro
    detections = detect_objects(frame)
    
    # Aquí puedes agregar código para procesar las detecciones
    
    # Muestra el cuadro procesado en Streamlit
    frame_container.image(frame, channels="BGR", use_column_width=True)
    
    # Rompe el bucle si el usuario presiona el botón 'Stop'
    if stop_button:
        break

cap.release()