import io
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, LSTM, Dense, Dropout
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import librosa.display


class VoiceRecognitionCNN:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.model_path = os.path.join('VoiceRecognition', 'models', 'voice_recognition_model.keras')
        self.classes_path = os.path.join('VoiceRecognition', 'models', 'classes.npy')
        self.input_shape = (100, 13)

        # Inicializar el LabelEncoder con clases vacías si no hay clases previamente
        if os.path.exists(self.classes_path):
            self.label_encoder.classes_ = np.load(self.classes_path, allow_pickle=True)
        else:
            self.label_encoder.fit([])  # Inicialmente sin etiquetas

    def add_new_user(self, new_user_label):
        current_classes = list(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else []
        current_classes.append(new_user_label)
        self.label_encoder.fit(current_classes)

    def initialize_model(self):
        np.random.seed(42)
        tf.random.set_seed(42)
        os.makedirs(os.path.join('VoiceRecognition', 'models'), exist_ok=True)

        if os.path.exists(self.model_path) and os.path.exists(self.classes_path):
            print("Loading pre-trained model and classes...")
            self.model = tf.keras.models.load_model(self.model_path)
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.load(self.classes_path, allow_pickle=True)
        else:
            print("Creating new model...")
            self.model = self.create_voice_recognition_model(self.input_shape, num_classes=1)
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.label_encoder = LabelEncoder()

    def create_voice_recognition_model(self, input_shape, num_classes):
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(128),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        return model

    def preprocess_audio(self, audio_data):
        # Convertir AudioData en un objeto de bytes
        audio_bytes = io.BytesIO(audio_data.get_wav_data())

        # Leer los datos de audio en un array NumPy usando librosa
        y, sr = librosa.load(audio_bytes, sr=16000)
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    def train_on_new_data(self, audio_data, user_label):
        mfccs = self.preprocess_audio(audio_data)

        # Verificar si el usuario ya está registrado en las clases
        if not hasattr(self.label_encoder, 'classes_') or user_label not in self.label_encoder.classes_:
            self.add_new_user(user_label)
            new_classes = np.append(self.label_encoder.classes_, user_label)
            self.label_encoder.classes_ = new_classes
            num_classes = len(new_classes)

            # Actualizar la capa final del modelo
            self.model.pop()  # Eliminar la última capa
            self.model.add(Dense(num_classes, activation='softmax', name=f'dense_{num_classes}'))  # Agregar nueva capa
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Convertir la etiqueta en formato categórico
        label = self.label_encoder.transform([user_label])
        label = to_categorical(label, num_classes=len(self.label_encoder.classes_))

        # Asegurarse de que el tamaño de las etiquetas coincida con el de las muestras de mfccs
        num_samples = mfccs.shape[1]  # Número de ventanas MFCC (frames)
        label_repeated = np.tile(label, (num_samples, 1))  # Repetir la etiqueta para cada ventana

        # Asegurarse de que el input tenga las dimensiones correctas para el modelo
        mfccs = np.expand_dims(mfccs, axis=0)  # Añadir una dimensión para que sea [batch_size, timesteps, features]

        # Entrenar el modelo
        if num_samples > 1:
            # Solo usamos validation_split si tenemos más de una muestra
            history = self.model.fit(mfccs, label_repeated, epochs=10, validation_split=0.2)
        else:
            # Entrenar sin validation_split si solo hay una muestra
            history = self.model.fit(mfccs, label_repeated, epochs=10)

        # Guardar el modelo y las clases
        self.model.save(self.model_path)
        np.save(self.classes_path, self.label_encoder.classes_)

        # Graficar el resultado
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if num_samples > 1:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        if num_samples > 1:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def recognize_command(self, audio_data):
        mfccs = self.preprocess_audio(audio_data)
        predictions = self.model.predict(mfccs)
        predicted_class = np.argmax(predictions)
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]

        print(f"Comando reconocido: {predicted_label}")
        return predicted_label
