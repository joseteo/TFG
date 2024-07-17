import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def create_voice_recognition_model(input_shape, num_classes):
    model = Sequential([
        # Convolutional layers
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # LSTM layers
        LSTM(128, return_sequences=True),
        Dropout(0.2),

        LSTM(128),
        Dropout(0.2),

        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(num_classes, activation='softmax')
    ])

    return model


# Generate dummy data (replace this with your actual data loading and preprocessing)
def generate_dummy_data(num_samples, time_steps, features, num_classes):
    X = np.random.randn(num_samples, time_steps, features)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    y = tf.keras.utils.to_categorical(y, num_classes)
    return X, y


# Set up parameters
time_steps = 100
features = 13
num_classes = 10
num_samples = 1000

# Generate dummy data
X_train, y_train = generate_dummy_data(num_samples, time_steps, features, num_classes)
X_val, y_val = generate_dummy_data(num_samples // 10, time_steps, features, num_classes)

# Create and compile the model
input_shape = (time_steps, features)
model = create_voice_recognition_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# Create the directory for saving the model if it doesn't exist
save_dir = os.path.join('VoiceRecognition', 'models')
os.makedirs(save_dir, exist_ok=True)

# Save the model
save_path = os.path.join(save_dir, 'voice_recognition_model.keras')
model.save(save_path)

print(f"Model saved to {save_path}")

# Create the assets directory if it doesn't exist
assets_dir = os.path.join('src', 'VoiceRecognition', 'assets')
os.makedirs(assets_dir, exist_ok=True)

# Plot training history
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

# Save the plot in the assets directory
plot_path = os.path.join(assets_dir, 'training_history.png')
plt.savefig(plot_path)
plt.close()

print(f"Training history plot saved to {plot_path}")