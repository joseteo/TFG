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
    images = []
    labels = []
    class_names = []
    class_dict = {}
    class_count = 0

    for person_name in os.listdir(data_path):
        person_path = os.path.join(data_path, person_name)
        if not os.path.isdir(person_path):
            continue

        if person_name not in class_dict:
            class_dict[person_name] = class_count
            class_names.append(person_name)
            class_count += 1

        label = class_dict[person_name]

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    images = images.reshape(-1, 64, 64, 1) / 255.0
    labels = to_categorical(labels, num_classes=len(class_names))

    return images, labels, class_names

# Cargar los datos
data_path = 'Reconocimiento Facial/Data'
X, y, class_names = load_data(data_path)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Configurar early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo
epochs = 50
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Guardar el modelo entrenado
model.save('Reconocimiento Facial/models/face_recognition_model.h5')
np.save('Reconocimiento Facial/models/class_names.npy', class_names)
