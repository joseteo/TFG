import threading
import cv2
import os
from queue import Queue
from tensorflow.keras.models import load_model
import numpy as np
from VoiceRecognition import process_voice
from VirtualMouse import process_hand
import screeninfo
from pynput.mouse import Controller, Button
import HandsTracking as sm

# Deshabilitar mensajes de advertencia de TensorFlow
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Cargar el modelo entrenado y los nombres de las clases
model = load_model('src/FacialRecognition/models/face_recognition_model.h5')
class_names = np.load('src/FacialRecognition/models/class_names.npy', allow_pickle=True)

# Inicializar variables globales
teseracto = False

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


def main_thread():
    cap = cv2.VideoCapture(0)

    # Iniciar hilo de reconocimiento de voz
    voice_thread = threading.Thread(target=process_voice, args=(voice_queue,))
    voice_thread.start()

    recognized_face = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se puede abrir la cámara")
            break

        # Procesar reconocimiento facial
        recognized_face = process_face(frame, face_queue, voice_queue)

        # Solo procesar manos y voz si se ha reconocido un rostro
        if recognized_face:
            frame = process_hand(frame)

        cv2.imshow("Proyecto", frame)
        if cv2.waitKey(1) == 27:  # Presiona 'Esc' para salir
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_thread()
