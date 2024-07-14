import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado y los nombres de las clases
model = load_model('Reconocimiento Facial/models/face_recognition_model.h5')  # Carga el modelo de reconocimiento facial entrenado
class_names = np.load('Reconocimiento Facial/models/class_names.npy', allow_pickle=True)  # Carga los nombres de las clases (nombres de las personas) desde un archivo

# Función para preprocesar la imagen
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convierte la imagen a escala de grises
    img = cv2.resize(img, (64, 64))  # Redimensiona la imagen a 64x64 píxeles
    img = img.reshape(1, 64, 64, 1) / 255.0  # Redimensiona y normaliza la imagen
    return img  # Retorna la imagen preprocesada

# Función para predecir el rostro
def predict_face(model, img):
    img = preprocess_image(img)  # Preprocesa la imagen
    predictions = model.predict(img)  # Realiza la predicción con el modelo
    predicted_class = np.argmax(predictions)  # Obtiene la clase predicha con la mayor probabilidad
    confidence = predictions[0][predicted_class]  # Obtiene la confianza de la predicción
    return class_names[predicted_class], confidence  # Retorna el nombre de la clase y la confianza

# Función para procesar el reconocimiento facial
def process_face(frame, face_queue, voice_queue):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        predicted_name, confidence = predict_face(model, face_img)
        if confidence > 0.6:  # Umbral de confianza
            cv2.putText(frame, f'{predicted_name} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            face_queue.put((predicted_name, confidence))
            return True
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return False

# Capturar video en tiempo real
cap = cv2.VideoCapture(0)  # Inicia la captura de video desde la cámara web (dispositivo 0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Carga el clasificador de Haar para detección de rostros

while True:  # Bucle infinito para procesar el video en tiempo real
    ret, frame = cap.read()  # Lee un cuadro de video
    if not ret:  # Si no se puede leer el cuadro, salir del bucle
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convierte el cuadro a escala de grises
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # Detecta rostros en el cuadro

    for (x, y, w, h) in faces:  # Itera sobre los rostros detectados
        face_img = frame[y:y+h, x:x+w]  # Recorta la imagen del rostro detectado
        predicted_name, confidence = predict_face(model, face_img)  # Predice el rostro utilizando el modelo
        if confidence > 0.6:  # Si la confianza de la predicción es mayor que 0.6
            cv2.putText(frame, f'{predicted_name} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)  # Muestra el nombre y la confianza en el cuadro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Dibuja un rectángulo alrededor del rostro detectado

    cv2.imshow('Reconocimiento Facial', frame)  # Muestra el cuadro con los resultados de reconocimiento

    if cv2.waitKey(1) == 27:  # Si se presiona la tecla 'Esc', salir del bucle
        break

cap.release()  # Libera la captura de video
cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas por OpenCV
