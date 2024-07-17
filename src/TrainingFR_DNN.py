import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Función para cargar los datos
def load_data(data_path):
    images = []  # Lista para almacenar las imágenes
    labels = []  # Lista para almacenar las etiquetas
    class_names = []  # Lista para almacenar los nombres de las clases
    class_dict = {}  # Diccionario para mapear nombres de clases a etiquetas
    class_count = 0  # Contador para las clases

    for person_name in os.listdir(data_path):  # Iterar sobre cada carpeta en el directorio de datos
        person_path = os.path.join(data_path, person_name)
        if not os.path.isdir(person_path):  # Verificar si es un directorio
            continue

        if person_name not in class_dict:  # Si la clase no está en el diccionario
            class_dict[person_name] = class_count  # Asignar un nuevo número de clase
            class_names.append(person_name)  # Agregar el nombre de la clase a la lista
            class_count += 1  # Incrementar el contador de clases

        label = class_dict[person_name]  # Obtener la etiqueta de la clase

        for img_name in os.listdir(person_path):  # Iterar sobre cada imagen en la carpeta de la persona
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)  # Leer la imagen
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir la imagen a escala de grises
            img = cv2.resize(img, (64, 64))  # Redimensionar la imagen a 64x64 píxeles
            images.append(img)  # Agregar la imagen a la lista de imágenes
            labels.append(label)  # Agregar la etiqueta a la lista de etiquetas

    images = np.array(images)  # Convertir la lista de imágenes a un array de NumPy
    labels = np.array(labels)  # Convertir la lista de etiquetas a un array de NumPy
    images = images.reshape(-1, 64, 64, 1) / 255.0  # Redimensionar y normalizar las imágenes
    labels = to_categorical(labels, num_classes=len(class_names))  # Convertir las etiquetas a una matriz categórica

    return images, labels, class_names  # Retornar las imágenes, etiquetas y nombres de las clases

# Cargar los datos
data_path = 'src/FacialRecognition/Data'
X, y, class_names = load_data(data_path)  # Cargar las imágenes y etiquetas desde el directorio de datos

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Dividir los datos en conjuntos de entrenamiento y prueba

# Crear el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),  # Primera capa convolucional
    MaxPooling2D((2, 2)),  # Primera capa de pooling
    Conv2D(64, (3, 3), activation='relu'),  # Segunda capa convolucional
    MaxPooling2D((2, 2)),  # Segunda capa de pooling
    Flatten(),  # Aplanar la salida de las capas anteriores
    Dense(128, activation='relu'),  # Capa densa con 128 unidades y activación ReLU
    Dense(len(class_names), activation='softmax')  # Capa de salida con tantas unidades como clases y activación softmax
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compilar el modelo con optimizador Adam y pérdida categórica

# Configurar early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)  # Configurar early stopping para detener el entrenamiento si la pérdida de validación no mejora durante 25 epochs

# Entrenar el modelo
epochs = 50  # Número de epochs
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping])  # Entrenar el modelo con los datos de entrenamiento y validación, usando early stopping

# Guardar el modelo entrenado
model.save('FacialRecognition/models/face_recognition_model.h5')  # Guardar el modelo entrenado en un archivo
np.save('FacialRecognition/models/class_names.npy', class_names)  # Guardar los nombres de las clases en un archivo
