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
def plot_training_history(history, output_dir, lr):
    # Crear las gráficas
    plt.figure(figsize=(12, 4))

    # Curva de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.title(f'Curva de Pérdida (LR={lr})')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Guardar la gráfica de pérdida
    loss_fig_path = os.path.join(output_dir, f'training_loss_lr_{lr}.png')
    plt.savefig(loss_fig_path)

    # Curva de precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de validación')
    plt.title(f'Curva de Precisión (LR={lr})')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    # Guardar la gráfica de precisión
    acc_fig_path = os.path.join(output_dir, f'training_accuracy_lr_{lr}.png')
    plt.savefig(acc_fig_path)

    # Mostrar las gráficas
    plt.show()


# Función para crear y entrenar el modelo
def create_and_train_model(X_train, y_train, X_test, y_test, class_names, lr, output_dir):
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

    # Compilar el modelo con una tasa de aprendizaje específica
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Configurar early stopping y checkpoints
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'best_model_lr_{lr}.keras', monitor='val_loss', save_best_only=True)

    # Entrenar el modelo
    epochs = 50
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping, checkpoint])

    # Guardar el modelo
    save_path = os.path.join(output_dir, f'face_recognition_model_lr_{lr}.keras')
    model.save(save_path)

    # Graficar el historial de entrenamiento
    plot_training_history(history, output_dir, lr)


# Cargar los datos
data_path = 'FacialRecognition\Data'
X, y, class_names = load_data(data_path)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el directorio para guardar el modelo y las gráficas si no existe
save_dir = os.path.join('FacialRecognition', 'models')
os.makedirs(save_dir, exist_ok=True)

# Probar diferentes tasas de aprendizaje
learning_rates = [0.001, 0.01, 0.1]

for lr in learning_rates:
    create_and_train_model(X_train, y_train, X_test, y_test, class_names, lr, save_dir)

















# # Ejemplo de datos ficticios para diferentes configuraciones de hiperparámetros
# results = {
#     'learning_rate_0.001': {'train_acc': 0.85, 'val_acc': 0.80},
#     'learning_rate_0.01': {'train_acc': 0.88, 'val_acc': 0.82},
#     'learning_rate_0.1': {'train_acc': 0.82, 'val_acc': 0.78},
#     'num_layers_2': {'train_acc': 0.87, 'val_acc': 0.83},
#     'num_layers_3': {'train_acc': 0.89, 'val_acc': 0.85},
#     'activation_relu': {'train_acc': 0.88, 'val_acc': 0.84},
#     'activation_tanh': {'train_acc': 0.86, 'val_acc': 0.81},
# }
#
# import matplotlib.pyplot as plt
#
# # Extraer los nombres de los hiperparámetros y las precisiones correspondientes
# labels = list(results.keys())
# train_acc = [results[label]['train_acc'] for label in labels]
# val_acc = [results[label]['val_acc'] for label in labels]
#
# # Crear la gráfica
# plt.figure(figsize=(10, 5))
# x = range(len(labels))
#
# plt.plot(x, train_acc, label='Precisión de entrenamiento', marker='o')
# plt.plot(x, val_acc, label='Precisión de validación', marker='o')
#
# plt.xticks(x, labels, rotation=45, ha='right')
# plt.xlabel('Configuraciones de Hiperparámetros')
# plt.ylabel('Precisión')
# plt.title('Impacto del Ajuste de Hiperparámetros en la Precisión del Modelo')
# plt.legend()
#
# # Guardar la gráfica
# plt.tight_layout()
# plt.savefig('hyperparameter_tuning.png')
# plt.show()
