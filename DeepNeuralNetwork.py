import numpy as np


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return Z > 0


def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=1, keepdims=True)


def cross_entropy_loss(Y, A):
    m = Y.shape[0]
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


# Datos de entrenamiento (usar datos adecuados según el caso)
X_train = np.random.randn(20, 100)  # 100 ejemplos con 20 características cada uno
Y_train = np.eye(3)[np.random.choice(3, 100)].T  # 100 ejemplos con 3 clases

# Definición de la red neuronal
layers_dims = [20, 64, 32, 3]  # Capa de entrada (20), dos capas ocultas (64 y 32), capa de salida (3)
nn = DeepNeuralNetwork(layers_dims)

# Entrenamiento del modelo
nn.train(X_train, Y_train, learning_rate=0.01, epochs=1000)
