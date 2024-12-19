# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:22:49 2024

@author: jperezr
"""

import os
import cv2
import numpy as np
import streamlit as st
import time

# Ruta local del archivo yolov3.weights
weights_path = "yolov3.weights"

# Verificar si el archivo yolov3.weights existe
if not os.path.exists(weights_path):
    st.error(f"El archivo {weights_path} no se encuentra en el directorio.")
    st.stop()

# Rutas a los archivos necesarios
cfg_path = "yolov3.cfg"
names_path = "coco.names"
video_path = "america_monterrey.mp4"  # Ruta del video cargado automáticamente

# Comprobar si los archivos necesarios existen
if not os.path.exists(cfg_path) or not os.path.exists(names_path):
    st.error("Faltan los archivos .cfg o .names. Asegúrate de que estén en el directorio correcto.")
    st.stop()

# Cargar el modelo YOLO
try:
    net = cv2.dnn.readNet(weights_path, cfg_path)
except cv2.error as e:
    st.error(f"Error al cargar el modelo YOLO: {e}")
    st.stop()

# Cargar las clases
try:
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    st.error("No se encontró el archivo coco.names.")
    st.stop()

# Función para procesar el video con detección
def process_video(video_path):
    # Usar OpenCV para capturar el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("No se pudo abrir el video.")
        st.stop()

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
                if confidence > 0.5:  # Umbral de confianza
                    # Coordenadas del cuadro delimitador
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    # Cuadro delimitador
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Solo dibujar cuadros alrededor de las personas (jugadores)
                    if classes[class_id] == "person":
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convertir la imagen de BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb  # Generamos cada fotograma para mostrar en Streamlit

    cap.release()

# Configurar la interfaz de Streamlit
st.title('Detección de Jugadores en Video')
st.write("Mostrando detección automática del video cargado.")

# Mostrar el video con detecciones de jugadores automáticamente
stframe = st.empty()

for frame in process_video(video_path):
    stframe.image(frame, channels="RGB", use_container_width=True)
    time.sleep(0.05)  # Controla la velocidad del video

st.write("\u2714\ufe0f \u00a1Detección completa!")
