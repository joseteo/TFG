import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow as tf


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


# Función para graficar las curvas de pérdida y precisión
def plot_training_history(history, output_dir, config_name):
    # Crear las gráficas
    plt.figure(figsize=(12, 4))

    # Curva de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.title(f'Curva de Pérdida ({config_name})')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Guardar la gráfica de pérdida
    loss_fig_path = os.path.join(output_dir, f'training_loss_{config_name}.png')
    plt.savefig(loss_fig_path)

    # Curva de precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de validación')
    plt.title(f'Curva de Precisión ({config_name})')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    # Guardar la gráfica de precisión
    acc_fig_path = os.path.join(output_dir, f'training_accuracy_{config_name}.png')
    plt.savefig(acc_fig_path)

    # Mostrar las gráficas
    plt.show()


# Función para crear y entrenar el modelo
def create_and_train_model(X_train, y_train, X_test, y_test, class_names, lr, num_layers, batch_size, activation, output_dir):
    # Crear el modelo
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation=activation, input_shape=(64, 64, 1)))
    model.add(MaxPooling2D((2, 2)))

    # Añadir más capas de convolución según num_layers
    for _ in range(1, num_layers):
        model.add(Conv2D(64, (3, 3), activation=activation))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dense(len(class_names), activation='softmax'))

    # Compilar el modelo con una tasa de aprendizaje específica
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Configurar early stopping y checkpoints
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'best_model_lr_{lr}_layers_{num_layers}_batch_{batch_size}_activation_{activation}.keras', monitor='val_loss', save_best_only=True)

    # Entrenar el modelo
    epochs = 50
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping, checkpoint])

    # Guardar el modelo
    save_path = os.path.join(output_dir, f'face_recognition_model_lr_{lr}_layers_{num_layers}_batch_{batch_size}_activation_{activation}.keras')
    model.save(save_path)

    # Graficar el historial de entrenamiento
    plot_training_history(history, output_dir, f'lr_{lr}_layers_{num_layers}_batch_{batch_size}_activation_{activation}')


# Cargar los datos
data_path = 'FacialRecognition/Data'
X, y, class_names = load_data(data_path)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el directorio para guardar el modelo y las gráficas si no existe
save_dir = os.path.join('FacialRecognition', 'models')
os.makedirs(save_dir, exist_ok=True)

# Listas de hiperparámetros a probar
learning_rates = [0.001, 0.01, 0.1]
num_layers_options = [2, 3]
batch_sizes = [16, 32]
activations = ['relu', 'tanh']

# Iterar sobre cada combinación de hiperparámetros
for lr in learning_rates:
    for num_layers in num_layers_options:
        for batch_size in batch_sizes:
            for activation in activations:
                create_and_train_model(X_train, y_train, X_test, y_test, class_names, lr, num_layers, batch_size, activation, save_dir)
