import threading
import cv2
import os
from queue import Queue
from tensorflow.keras.models import load_model
import numpy as np
from VoiceRecognition import VoiceRecognitionSystem  
from VirtualMouse import process_hand
import screeninfo
from pynput.mouse import Controller, Button
import HandsTracking as sm
import pygetwindow as gw
from CaptureWindow import capture_window_continuous
from PCServer import process_and_merge_frames

# Deshabilitar mensajes de advertencia de TensorFlow
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Cargar el modelo entrenado y los nombres de las clases
model = load_model('src/FacialRecognition/models/face_recognition_model.h5')
class_names = np.load('src/FacialRecognition/models/class_names.npy', allow_pickle=True)

# Inicializar variables globales
teseracto = False
recognized_user = None  # Variable para almacenar el nombre del usuario reconocido
system_active = False  # Variable para indicar si el sistema está activo

# Colas para la comunicación entre hilos
face_queue = Queue()
voice_queue = Queue()


# Función para preprocesar la imagen
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = img.reshape(1, 64, 64, 1) / 255.0
    return img


# Función para predecir el rostro
def predict_face(model, img):
    img = preprocess_image(img)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]
    return class_names[predicted_class], confidence


# Función para procesar el reconocimiento facial
def process_face(frame, face_queue, voice_queue):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    recognized_face = False

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        predicted_name, confidence = predict_face(model, face_img)
        if confidence > 0.6:  # Umbral de confianza
            cv2.putText(frame, f'{predicted_name} ({confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (36, 255, 12), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            recognized_face = True
            face_queue.put((predicted_name, confidence))
            voice_queue.put((predicted_name, confidence))

    return recognized_face

# Función para procesar los marcadores ArUco y superponer la pantalla capturada
def process_aruco_markers(client_frame, pc_frame):
    # Detectar marcadores ArUco
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(client_frame, aruco_dict, parameters=aruco_params)

    if len(corners) > 0:
        # Si se detectan marcadores, procesar la superposición de la pantalla capturada
        client_frame = cv2.aruco.drawDetectedMarkers(client_frame, corners, ids)
        client_frame = process_and_merge_frames(client_frame, pc_frame, corners)

    return client_frame

# Función para cerrar la sesión del usuario
def logout_user():
    global recognized_user, system_active
    recognized_user = None
    system_active = False
    print("Sesión cerrada. Sistema inactivo.")

def main_thread():
    global recognized_user, system_active

    cap = cv2.VideoCapture(0)
    recognized_face = False

    # Capturar el título de la ventana activa
    active_window_title = gw.getActiveWindow().title

    # Iniciar la captura de pantalla del PC en un hilo separado
    pc_capture_gen = capture_window_continuous(active_window_title)
    pc_frame = next(pc_capture_gen)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se puede abrir la cámara")
            break

        # Si el sistema no está activo, intentar reconocer el rostro
        if not system_active:
            predicted_name, recognized_face = process_face(frame, face_queue, voice_queue)
            if recognized_face:
                recognized_user = predicted_name
                system_active = True
                print(f"Usuario {recognized_user} reconocido. Sistema activo.")

                # Iniciar el reconocimiento de voz para el usuario reconocido
                voice_system = VoiceRecognitionSystem(username=recognized_user)
                voice_thread = threading.Thread(target=voice_system.process_voice, args=(voice_queue,))
                voice_thread.start()
        
        # Procesar manos, voz y marcadores ArUco solo si el sistema está activo
        if system_active:
            frame = process_hand(frame)
            frame = process_aruco_markers(frame, pc_frame)

            # Verificar si se ha dado el comando de voz para cerrar sesión
            voice_input = voice_queue.get() if not voice_queue.empty() else None
            if voice_input and "cerrar sesión" in voice_input[0].lower() and recognized_user in voice_input[0].lower():
                logout_user()

        cv2.imshow("Proyecto", frame)
        if cv2.waitKey(1) == 27:  # Presiona 'Esc' para salir
            break

        # Capturar el siguiente frame del PC para la superposición
        pc_frame = next(pc_capture_gen)

    cap.release()
    cv2.destroyAllWindows()




    # cap = cv2.VideoCapture(0)
    # recognized_face = False

    # # Iniciar hilo de reconocimiento de voz
    # voice_thread = threading.Thread(target=process_voice, args=(voice_queue,))
    # voice_thread.start()

    # # Capturar el título de la ventana activa
    # active_window_title = gw.getActiveWindow().title

    # # Iniciar la captura de pantalla del PC en un hilo separado
    # pc_capture_gen = capture_window_continuous(active_window_title)
    # pc_frame = next(pc_capture_gen)

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("No se puede abrir la cámara")
    #         break

    #     # Procesar reconocimiento facial
    #     recognized_face = process_face(frame, face_queue, voice_queue)

    #     # Solo procesar manos, voz y marcadores ArUco si se ha reconocido un rostro
    #     if recognized_face:
    #         frame = process_hand(frame)
    #         frame = process_aruco_markers(frame, pc_frame)

    #     cv2.imshow("Proyecto", frame)
    #     if cv2.waitKey(1) == 27:  # Presiona 'Esc' para salir
    #         break

    #     # Capturar el siguiente frame del PC para la superposición
    #     pc_frame = next(pc_capture_gen)

    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main_thread()
