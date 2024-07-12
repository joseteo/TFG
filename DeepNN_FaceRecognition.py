import numpy as np
import cv2
import os

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

def cross_entropy_loss(Y, A):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(A + 1e-9)) / m
    return loss

class DeepNeuralNetwork:
    def __init__(self, layers_dims):
        np.random.seed(1)
        self.parameters = {}
        self.L = len(layers_dims) - 1
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    def forward_propagation(self, X):
        self.caches = {}
        A = X
        self.caches['A0'] = A
        for l in range(1, self.L):
            Z = np.dot(self.parameters['W' + str(l)], A) + self.parameters['b' + str(l)]
            A = relu(Z)
            self.caches['Z' + str(l)] = Z
            self.caches['A' + str(l)] = A
        ZL = np.dot(self.parameters['W' + str(self.L)], A) + self.parameters['b' + str(self.L)]
        AL = softmax(ZL)
        self.caches['Z' + str(self.L)] = ZL
        self.caches['A' + str(self.L)] = AL
        return AL

    def backward_propagation(self, X, Y):
        m = X.shape[1]
        grads = {}
        AL = self.caches['A' + str(self.L)]
        dZL = AL - Y
        grads['dW' + str(self.L)] = 1. / m * np.dot(dZL, self.caches['A' + str(self.L - 1)].T)
        grads['db' + str(self.L)] = 1. / m * np.sum(dZL, axis=1, keepdims=True)
        dA_prev = np.dot(self.parameters['W' + str(self.L)].T, dZL)

        for l in reversed(range(1, self.L)):
            dZ = dA_prev * relu_derivative(self.caches['Z' + str(l)])
            grads['dW' + str(l)] = 1. / m * np.dot(dZ, self.caches['A' + str(l - 1)].T)
            grads['db' + str(l)] = 1. / m * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dA_prev = np.dot(self.parameters['W' + str(l)].T, dZ)

        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    def train(self, X, Y, learning_rate=0.01, epochs=1000):
        for epoch in range(epochs):
            AL = self.forward_propagation(X)
            grads = self.backward_propagation(X, Y)
            self.update_parameters(grads, learning_rate)
            if epoch % 100 == 0:
                loss = cross_entropy_loss(Y, AL)
                print(f"Epoch {epoch}, Loss: {loss}")

def load_data(data_path, img_size):
    """
    Carga los datos desde el directorio especificado y devuelve las imágenes y las etiquetas.
    """
    X = []
    Y = []
    class_names = os.listdir(data_path)
    class_indices = {class_name: i for i, class_name in enumerate(class_names)}
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            X.append(img)
            Y.append(class_indices[class_name])
    X = np.array(X).reshape(len(X), -1).T / 255.0  # Normalización y reshaping
    Y = np.eye(len(class_names))[Y].T  # One-hot encoding de las etiquetas
    return X, Y, class_names

# Ruta de los datos de entrenamiento
data_path = 'path_to_your_face_data'

# Tamaño de las imágenes de entrada (usar el tamaño adecuado)
img_size = (64, 64)

# Cargar los datos de entrenamiento
X_train, Y_train, class_names = load_data(data_path, img_size)

# Definición de la red neuronal
layers_dims = [X_train.shape[0], 128, 64, len(class_names)]  # Capa de entrada, capas ocultas y capa de salida
nn = DeepNeuralNetwork(layers_dims)

# Entrenamiento del modelo
nn.train(X_train, Y_train, learning_rate=0.01, epochs=1000)

# Guardar el modelo entrenado
np.save('face_recognition_parameters.npy', nn.parameters)

# Función para predecir
def predict_face(nn, X):
    AL = nn.forward_propagation(X)
    return np.argmax(AL, axis=0)

# Ejemplo de uso
# Supongamos que tienes una imagen nueva para predecir
new_img_path = 'path_to_new_face_image'
new_img = cv2.imread(new_img_path)
new_img = cv2.resize(new_img, img_size).reshape(-1, 1) / 255.0

prediction = predict_face(nn, new_img)
predicted_class = class_names[prediction[0]]
print(f'Predicted class: {predicted_class}')
