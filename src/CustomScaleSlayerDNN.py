import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Definir la capa personalizada
class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.multiply(inputs, self.scale)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'scale': self.scale})
        return config

# Registrar la capa personalizada
tf.keras.utils.get_custom_objects().update({'CustomScaleLayer': CustomScaleLayer})

# Cargar el modelo entrenado y los nombres de las clases
model = load_model('Reconocimiento Facial/models/face_recognition_model.h5', custom_objects={'CustomScaleLayer': CustomScaleLayer})
class_names = np.load('Reconocimiento Facial/models/class_names.npy', allow_pickle=True)

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

# Capturar video en tiempo real
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        predicted_name, confidence = predict_face(model, face_img)
        if confidence > 0.6:  # Umbral de confianza
            cv2.putText(frame, f'{predicted_name} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Reconocimiento Facial', frame)

    if cv2.waitKey(1) == 27:  # Presiona 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()
