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
        self.label_encoder = None
        self.model_path = os.path.join('VoiceRecognition', 'models', 'voice_recognition_model.keras')
        self.classes_path = os.path.join('VoiceRecognition', 'models', 'classes.npy')
        self.input_shape = (100, 13)  # Adjust as necessary

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
        y, sr = librosa.load(audio_data, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()

        mfccs = np.expand_dims(mfccs, axis=0)
        return mfccs

    def train_on_new_data(self, audio_data, user_label):
        mfccs = self.preprocess_audio(audio_data)

        if user_label not in self.label_encoder.classes_:
            new_classes = np.append(self.label_encoder.classes_, user_label)
            self.label_encoder.classes_ = new_classes
            num_classes = len(new_classes)
            self.model.layers[-1] = Dense(num_classes, activation='softmax')
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        label = self.label_encoder.transform([user_label])
        label = to_categorical(label, num_classes=len(self.label_encoder.classes_))

        history = self.model.fit(mfccs, label, epochs=10, validation_split=0.2)

        self.model.save(self.model_path)
        np.save(self.classes_path, self.label_encoder.classes_)

        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
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
        return predicted_label




# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, LSTM, Dense, Dropout
# import matplotlib.pyplot as plt
# import librosa
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.utils import to_categorical

# # Set random seed for reproducibility
# np.random.seed(42)
# tf.random.set_seed(42)


# def create_voice_recognition_model(input_shape, num_classes):
#     model = Sequential([
#         # Convolutional layers
#         Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
#         BatchNormalization(),
#         MaxPooling1D(pool_size=2),

#         Conv1D(64, kernel_size=3, activation='relu'),
#         BatchNormalization(),
#         MaxPooling1D(pool_size=2),

#         Conv1D(128, kernel_size=3, activation='relu'),
#         BatchNormalization(),
#         MaxPooling1D(pool_size=2),

#         # LSTM layers
#         LSTM(128, return_sequences=True),
#         Dropout(0.2),

#         LSTM(128),
#         Dropout(0.2),

#         # Dense layers
#         Dense(64, activation='relu'),
#         Dropout(0.2),

#         Dense(num_classes, activation='softmax')
#     ])

#     return model


# def preprocess_audio(file_path):
#     y, sr = librosa.load(file_path, sr=16000)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     mfccs = np.expand_dims(mfccs, axis=0)
#     return mfccs


# def capture_audio():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Listening...")
#         audio = recognizer.listen(source)
#     return audio


# # Load the pre-trained model
# def load_voice_recognition_model(model_path, input_shape, num_classes):
#     model = create_voice_recognition_model(input_shape, num_classes)
#     model.load_weights(model_path)
#     return model


# # Define paths
# model_path = os.path.join('VoiceRecognition', 'models', 'voice_recognition_model.keras')
# input_shape = (100, 13)  # This should match the input shape used during training
# num_classes = 10  # This should match the number of classes used during training

# # Load the model
# voice_model = load_voice_recognition_model(model_path, input_shape, num_classes)

# # Label Encoder
# label_encoder = LabelEncoder()
# label_encoder.classes_ = np.load('VoiceRecognition/models/classes.npy', allow_pickle=True)


# def recognize_command(model, audio):
#     mfccs = preprocess_audio(audio)
#     predictions = model.predict(mfccs)
#     predicted_class = np.argmax(predictions)
#     predicted_label = label_encoder.inverse_transform([predicted_class])[0]
#     return predicted_label


# # Generate dummy data (replace this with your actual data loading and preprocessing)
# def generate_dummy_data(num_samples, time_steps, features, num_classes):
#     X = np.random.randn(num_samples, time_steps, features)
#     y = np.random.randint(0, num_classes, size=(num_samples,))
#     y = tf.keras.utils.to_categorical(y, num_classes)
#     return X, y


# # Set up parameters
# time_steps = 100
# features = 13
# num_classes = 10
# num_samples = 1000

# # Generate dummy data
# X_train, y_train = generate_dummy_data(num_samples, time_steps, features, num_classes)
# X_val, y_val = generate_dummy_data(num_samples // 10, time_steps, features, num_classes)

# # Create and compile the model
# input_shape = (time_steps, features)
# model = create_voice_recognition_model(input_shape, num_classes)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Print model summary
# model.summary()

# # Train the model
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=10,
#     batch_size=32
# )

# # Create the directory for saving the model if it doesn't exist
# save_dir = os.path.join('VoiceRecognition', 'models')
# os.makedirs(save_dir, exist_ok=True)

# # Save the model
# save_path = os.path.join(save_dir, 'voice_recognition_model.keras')
# model.save(save_path)

# print(f"Model saved to {save_path}")

# # Create the assets directory if it doesn't exist
# assets_dir = os.path.join('src', 'VoiceRecognition', 'assets')
# os.makedirs(assets_dir, exist_ok=True)

# # Plot training history
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()

# # Save the plot in the assets directory
# plot_path = os.path.join(assets_dir, 'training_history.png')
# plt.savefig(plot_path)
# plt.close()

# print(f"Training history plot saved to {plot_path}")
