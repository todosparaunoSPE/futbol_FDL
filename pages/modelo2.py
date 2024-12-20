# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:56:33 2024

@author: jperezr
"""

import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import streamlit as st

# Mostrar la sección de ayuda en el sidebar
st.sidebar.write("### Ayuda")
st.sidebar.write("""
Este código utiliza un modelo de detección de objetos basado en **EfficientDet** de TensorFlow Hub, 
el cual permite identificar y localizar objetos en imágenes o videos en tiempo real.

**¿Qué hace este código?**
- El código carga un modelo preentrenado para detección de objetos desde TensorFlow Hub.
- Se procesan los fotogramas de un video y se detectan los objetos en cada uno de ellos.
- El modelo realiza la predicción sobre los objetos presentes en los fotogramas.
- Los resultados se muestran en un contenedor de imágenes interactivo utilizando Streamlit.

**Modelo Utilizado:**
- El modelo usado es el **EfficientDet D0** de TensorFlow Hub, que es eficiente en términos de precisión y velocidad.
- El modelo está optimizado para realizar tareas de detección de objetos en imágenes de tamaño reducido (320x320 px).

**Autor:**
- **Javier Horacio Pérez Ricárdez**
""")

# Cargar el modelo desde TensorFlow Hub
model_url = "https://tfhub.dev/tensorflow/efficientdet/d0/1"  # Ejemplo de modelo de detección de objetos
model = hub.load(model_url)

# Función para preprocesar la imagen
def preprocess_image(image):
    # Redimensiona la imagen a una resolución más pequeña para acelerar el procesamiento
    image_resized = cv2.resize(image, (320, 320))  # Usar una resolución más baja
    
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

frame_counter = 0  # Contador de fotogramas procesados

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Procesa solo cada N fotogramas
    if frame_counter % 3 == 0:  # Procesar cada 3 fotogramas
        # Realiza la detección en el cuadro
        detections = detect_objects(frame)
        
        # Aquí puedes agregar código para procesar las detecciones
        
        # Muestra el cuadro procesado en Streamlit
        frame_container.image(frame, channels="BGR", use_container_width=True)  # Usar use_container_width en lugar de use_column_width
    
    frame_counter += 1

cap.release()
