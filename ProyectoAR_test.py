import os
import cv2
import numpy as np
import re
import pyttsx3
import speech_recognition as sr
import threading
import queue
from pynput.mouse import Controller, Button
import screeninfo
from SeguimientoManos import detectormanos
from RatonVirtual import process_hand

# Crear colas para la comunicación entre hebras
voice_queue = queue.Queue()
face_queue = queue.Queue()

# Variables globales
modo_r_d = False
circulo = False
rectangulo = False
rostro_reconocido = False

# Inicializar reconocimiento facial
dataPath = 'C:/Users/joset/PycharmProjects/pythonProject/Reconocimiento Facial/Data'
os.makedirs(dataPath, exist_ok=True)  # Crear el directorio si no existe
imagePaths = os.listdir(dataPath)  # Listar todas las personas en la base de datos
face_recognizer = cv2.face.EigenFaceRecognizer_create()  # Crear el reconocedor facial
face_recognizer.read('modeloReconocimientoFacial.xml')  # Cargar el modelo de reconocimiento facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Cargar el clasificador de detección de rostros

# Inicializar reconocimiento de voz
rec = sr.Recognizer()
mic = sr.Microphone()
engine = pyttsx3.init()
voices = engine.getProperty('voices')

def process_face(frame, face_queue, voice_queue):
    """
    Función para procesar el reconocimiento facial.
    """
    global rostro_reconocido
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)  # Convertir el frame a escala de grises
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)  # Detectar rostros en la imagen

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)  # Predecir la identidad del rostro
        face_queue.put(result)  # Colocar el resultado en la cola de rostros
        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if result[1] < 5700:  # Si la predicción es confiable
            person_name = imagePaths[result[0]]
            voice_queue.put((result, person_name))  # Colocar el nombre en la cola de voz
            cv2.putText(frame, '{}'.format(person_name), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rostro_reconocido = True  # Indicar que se ha reconocido un rostro
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            rostro_reconocido = False  # Indicar que no se ha reconocido un rostro

def process_voice(voice_queue):
    """
    Función para procesar el reconocimiento de voz.
    """
    global modo_r_d, circulo, rectangulo
    while True:
        result, person_name = voice_queue.get()
        try:
            with mic as source:
                rec.adjust_for_ambient_noise(source, duration=0.5)
                audio = rec.listen(source)  # Escuchar el audio del micrófono

                texto = rec.recognize_google(audio, language='es-ES')  # Reconocer el texto del audio

                if re.search(r"Alfred", texto, re.IGNORECASE):
                    print(f'\n Hola {person_name}')

                    engine.setProperty('rate', 160)
                    engine.setProperty('voice',
                                       'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ES-ES_HELENA_11.0')
                    engine.say(f'Hola {person_name}')
                    engine.runAndWait()

                    if re.search(r"Ratón|ratón|Raton|raton", texto, re.IGNORECASE):
                        print('Seleccionada Mano Raton Virtual')
                        modo_r_d = False
                    elif re.search(r"Dibujar|circulo", texto, re.IGNORECASE) and (result[0] == 0 or result[0] == 2):
                        modo_r_d = True
                        circulo = True
                        rectangulo = False
                        print('Ha seleccionado dibujar la forma circulo')
                    elif re.search(r"Dibujar|rectangle", texto, re.IGNORECASE) and (result[0] == 0 or result[0] == 2):
                        modo_r_d = True
                        rectangulo = True
                        circulo = False
                        print('Ha seleccionado dibujar la forma rectangulo')

        except sr.RequestError:
            print("API no disponible")
        except sr.UnknownValueError:
            print("")

def main_thread():
    """
    Función principal que ejecuta el hilo principal de la aplicación.
    """
    global rostro_reconocido
    cap = cv2.VideoCapture(0)  # Iniciar captura de video

    voice_thread = threading.Thread(target=process_voice, args=(voice_queue,))  # Iniciar el hilo para el reconocimiento de voz
    voice_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se puede abrir la cámara")
            break

        # Primero, intentamos reconocer un rostro
        process_face(frame, face_queue, voice_queue)

        # Si se ha reconocido un rostro, entonces procesamos las manos
        if rostro_reconocido:
            frame = process_hand(frame)  # Procesar reconocimiento de manos

        cv2.imshow("Proyecto", frame)  # Mostrar el frame en la ventana
        if cv2.waitKey(1) == 27:  # Salir si se presiona la tecla ESC
            break

    cap.release()  # Liberar la captura de video
    cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV

if __name__ == "__main__":
    main_thread()  # Ejecutar la función principal