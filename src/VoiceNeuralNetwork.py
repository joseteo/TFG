import numpy as np
import speech_recognition as sr
import pyttsx3
import pyautogui
import os
import librosa
import webbrowser
import tempfile


# Red Neuronal Básica
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))

    def feedforward(self, x):
        self.hidden = sigmoid(np.dot(x, self.weights1) + self.bias1)
        self.output = sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output

    def backpropagation(self, x, y, output):
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)
        self.weights2 += self.hidden.T.dot(output_delta)
        self.weights1 += x.T.dot(hidden_delta)
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True)
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, x, y, epochs=1000):
        for epoch in range(epochs):
            output = self.feedforward(x)
            self.backpropagation(x, y, output)
            if epoch % 100 == 0:
                loss = mse_loss(y, output)
                print(f"Epoch {epoch}, Loss: {loss}")

    def save_model(self, file_name):
        np.savez(file_name, weights1=self.weights1, weights2=self.weights2, bias1=self.bias1, bias2=self.bias2)

    def load_model(self, file_name):
        npzfile = np.load(file_name)
        self.weights1 = npzfile['weights1']
        self.weights2 = npzfile['weights2']
        self.bias1 = npzfile['bias1']
        self.bias2 = npzfile['bias2']


# Funciones de activación y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Función de pérdida
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


# Inicialización de la red neuronal
nn = NeuralNetwork(input_size=20, hidden_size=10, output_size=3)  # Ajustar tamaños según características MFCC

# Inicializa el motor de síntesis de voz
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 150)


def speak(text):
    engine.say(text)
    engine.runAndWait()


def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        try:
            query = r.recognize_google(audio, language='en-US')
            print(f"User said: {query}\n")
        except sr.UnknownValueError:
            speak("Sorry, I did not understand that.")
            return "None"
        except sr.RequestError:
            speak("Sorry, my speech service is down.")
            return "None"
        return query.lower()


def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfccs.T, axis=0)


def execute_command(command):
    if command == 0:
        speak("Opening Notepad")
        os.system('notepad')
    elif command == 1:
        speak("Opening Browser")
        webbrowser.open('http://google.com')
    elif command == 2:
        speak("Shutting down the system")
        os.system('shutdown /s /t 1')
    else:
        speak("Sorry, I don't know that command.")


def main():
    try:
        nn.load_model('nn_model.npz')
        print("Modelo cargado exitosamente.")
    except FileNotFoundError:
        print("No se encontró el modelo, entrenando desde cero.")

    speak("Hello, how can I assist you?")
    while True:
        audio_file = listen()
        if audio_file == "none":
            continue

        # Guardar el audio capturado en un archivo temporal
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
            librosa.output.write_wav(temp_audio_file.name, audio_file, sr=16000)
            features = extract_features(temp_audio_file.name)

        features = features.reshape(1, -1)
        command = np.argmax(nn.feedforward(features))
        execute_command(command)

        # Datos de entrenamiento continuo (suponiendo que el usuario pueda confirmar la acción correcta)
        x_new = features
        y_new = np.zeros((1, 3))  # Etiquetas en one-hot encoding
        y_new[0, command] = 1

        nn.train(x_new, y_new, epochs=1000)
        nn.save_model('nn_model.npz')

        if 'exit' in audio_file or 'quit' in audio_file:
            speak("Goodbye!")
            break


if __name__ == "__main__":
    main()
