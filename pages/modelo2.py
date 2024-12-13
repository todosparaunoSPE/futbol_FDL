# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:56:33 2024

@author: jperezr
"""

import streamlit as st
import cv2
import numpy as np
import time
import os  # Para verificar si el archivo existe

# Configuración de la aplicación
st.title("Análisis de Fuera de Lugar en Fútbol")

# Parámetros de control
frame_rate = st.sidebar.slider("Procesar 1 de cada n cuadros", 1, 20, 5)  # Frecuencia de análisis
delay = st.sidebar.slider("Retraso entre cuadros (segundos)", 0.1, 5.0, 1.0)  # Retraso intencional

# Sección de ayuda visible automáticamente en la barra lateral
st.sidebar.markdown("""
### ¿Qué hace esta aplicación?

Esta aplicación está diseñada para analizar un video de fútbol y detectar posibles líneas en el campo, ayudando en la revisión de jugadas para determinar si existe un fuera de lugar. 

La aplicación se centra en la **jugada del gol del América al Monterrey** durante el partido de ida de la final jugado el **12 de diciembre de 2024** en **Puebla, Puebla**. El objetivo principal es proporcionar una herramienta visual para revisar si la jugada que resultó en el gol fue legítima o si hubo algún fuera de lugar.

### Funcionalidades:
1. **Análisis del video**: 
    - Carga y procesa un video de fútbol (en este caso, el video de la jugada de gol entre América y Monterrey).
    - Analiza las líneas del campo utilizando procesamiento de imágenes con OpenCV para determinar posibles posiciones de fuera de lugar.
    
2. **Revisión en tiempo real**:
    - Visualiza el video procesado en tiempo real, mostrando las líneas detectadas en el campo.
    - Permite verificar si la jugada fue legal, basado en las posiciones de los jugadores y las líneas del campo.
    
3. **Parámetros ajustables**:
    - Ajusta la cantidad de cuadros analizados por segundo, para un análisis más detallado o más rápido.
    - Controla el retraso entre cuadros para observar la jugada a una velocidad cómoda.

### ¿Cómo usarla?
1. **Carga un video**: Puedes usar el archivo predeterminado de la jugada del gol, o cargar un video propio.
2. **Ajusta los parámetros**: Modifica la frecuencia de análisis y el retraso entre cuadros a través de los controles de la barra lateral.
3. **Observa el análisis**: El video se mostrará con las líneas detectadas y podrás ver si la jugada fue válida o si se produjo un fuera de lugar.

### Notas:
- Esta herramienta es una simulación básica utilizando procesamiento de imágenes, y no garantiza una detección exacta en situaciones complejas.
- Para obtener un análisis más preciso, podrían integrarse algoritmos más avanzados de visión por computadora o inteligencia artificial.

### Desarrollado por:
**Javier Horacio Pérez Ricárdez**
""")

# Nombre del archivo de video
video_file = "america_monterrey.mp4"
uploaded_video = None

# Verificar si el video existe localmente
if os.path.exists(video_file):
    st.sidebar.success(f"Video '{video_file}' cargado automáticamente.")
    uploaded_video = video_file
else:
    st.sidebar.header("Subir un Video")
    uploaded_video = st.sidebar.file_uploader("Carga un video en formato MP4", type=["mp4", "avi", "mov"])

if uploaded_video:
    st.sidebar.success("Video cargado con éxito!")
    stframe = st.empty()

    # Si es un archivo subido, guardarlo temporalmente
    if not isinstance(uploaded_video, str):  # Si no es una cadena (archivo local), es un archivo subido
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
    else:
        video_path = uploaded_video  # Usar el archivo local

    # Procesar el video con OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_counter = 0  # Contador para controlar los cuadros analizados

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar solo 1 de cada n cuadros
        frame_counter += 1
        if frame_counter % frame_rate != 0:
            continue

        # Detección de bordes (simplificada)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detectar líneas del campo
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=5)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Mostrar el frame procesado en Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)  # Usar use_container_width en lugar de use_column_width

        # Agregar un retraso intencional
        time.sleep(delay)

    cap.release()

st.sidebar.info("Cuando termines, cierra la aplicación.")
