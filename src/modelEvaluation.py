import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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
data_path = 'FacialRecognition\Data'
X, y, class_names = load_data(data_path)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Si el modelo ya ha sido entrenado y guardado, puedes cargarlo en lugar de entrenarlo de nuevo.
model_path = os.path.join('FacialRecognition', 'models', 'face_recognition_model.keras')

if os.path.exists(model_path):
    # Cargar el modelo ya entrenado
    model = load_model(model_path)
else:
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

    # Configurar early stopping y checkpoints
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

    # Entrenar el modelo
    epochs = 50
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping, checkpoint])

    # Guardar el modelo
    save_dir = os.path.join('FacialRecognition', 'models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'face_recognition_model.keras')
    model.save(save_path)

# Evaluar el modelo en el conjunto de prueba
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calcular métricas
accuracy = accuracy_score(y_true, y_pred_classes)
report = classification_report(y_true, y_pred_classes, target_names=class_names)
cm = confusion_matrix(y_true, y_pred_classes)

# Mostrar los resultados
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Graficar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Guardar la gráfica
evaluation_fig_path = os.path.join('FacialRecognition', 'models', 'model_evaluation.png')
plt.savefig(evaluation_fig_path)

# Mostrar la gráfica
plt.show()
