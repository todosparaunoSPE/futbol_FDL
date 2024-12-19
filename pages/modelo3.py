# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:22:49 2024

@author: jperezr
"""

import cv2
import numpy as np
import streamlit as st
import time
import os
import urllib.request

# Función para descargar yolov3.weights
def download_weights():
    url = "https://pjreddie.com/media/files/yolov3.weights"  # Enlace oficial de YOLO
    output_path = "yolov3.weights"
    
    if not os.path.exists(output_path):
        st.write("Descargando yolov3.weights, esto puede tomar algunos minutos...")
        urllib.request.urlretrieve(url, output_path)
        st.write("Descarga completada.")
    else:
        st.write("El archivo yolov3.weights ya está disponible.")

# Llama a la función antes de cargar el modelo
download_weights()

# Rutas a los archivos descargados
cfg_path = "yolov3.cfg"
weights_path = "yolov3.weights"
names_path = "coco.names"

# Verificar que los archivos necesarios estén presentes
if not os.path.exists(cfg_path) or not os.path.exists(names_path):
    st.error("Faltan archivos. Asegúrate de tener 'yolov3.cfg' y 'coco.names' en el directorio.")

# Cargar el modelo YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)

# Cargar las clases
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Función para procesar el video con detección
def process_video(video_path):
    # Usar OpenCV para capturar el video
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        
        # Preparar la imagen para YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # Ejecutar la detección
        detections = net.forward(output_layers)

        # Procesar las detecciones
        for out in detections:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Umbral de confianza (puedes ajustarlo)
                    # Coordenadas del cuadro delimitador
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    # Cuadro delimitador
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Solo dibujar cuadros alrededor de las personas (jugadores)
                    if classes[class_id] == "person":  # Asegúrate de que sea una persona
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convertir la imagen de BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb  # Generamos cada fotograma para mostrar en Streamlit

    cap.release()

# Configurar la interfaz de Streamlit
st.title('Detección de Jugadores en Video')
st.write("Sube un video de fútbol para detectar jugadores en él.")

# Subir un archivo de video
uploaded_video = st.file_uploader("Cargar video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Guardar el video temporalmente
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Mostrar el video con detecciones de jugadores
    stframe = st.empty()

    for frame in process_video(video_path):
        stframe.image(frame, channels="RGB", use_container_width=True)
        time.sleep(0.05)  # Controla la velocidad del video

    st.write("¡Detección completa!")
