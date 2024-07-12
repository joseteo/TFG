# Proyecto de Reconocimiento Facial con Red Neuronal Profunda

## Descripción

Este proyecto implementa un sistema de reconocimiento facial utilizando una red neuronal profunda (Deep Neural Network, DNN). El objetivo es crear una herramienta robusta capaz de identificar personas a partir de imágenes de rostros mediante técnicas avanzadas de procesamiento de imágenes y aprendizaje automático.

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Requisitos](#requisitos)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Preparación de Datos](#preparación-de-datos)
5. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
6. [Uso del Modelo](#uso-del-modelo)
7. [Evaluación y Resultados](#evaluación-y-resultados)
8. [Contribuciones](#contribuciones)
9. [Licencia](#licencia)

## Introducción

Este proyecto forma parte de mi Trabajo de Fin de Grado (TFG) y está diseñado para demostrar el uso de redes neuronales profundas en el reconocimiento facial. El sistema está compuesto por varios módulos que incluyen la captura de imágenes, la detección de rostros, el procesamiento de imágenes, y el reconocimiento facial mediante una DNN.

## Requisitos

Para ejecutar este proyecto, necesitas instalar las siguientes dependencias:

- Python 3.7+
- NumPy
- OpenCV
- TensorFlow / Keras
- PyAudio
- Pyttsx3
- Pynput
- Screeninfo

Puedes instalar las dependencias utilizando `pip`:

```bash
pip install numpy opencv-python tensorflow keras pyaudio pyttsx3 pynput screeninfo
```

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

.
├── data/                         # Directorio de datos
├── models/                       # Directorio para guardar modelos entrenados
├── src/                          # Código fuente
│   ├── main.py                   # Archivo principal
│   ├── hand_processing.py        # Módulo de procesamiento de manos
│   ├── deep_neural_network.py    # Implementación de la red neuronal profunda
│   ├── face_recognition.py       # Funciones de reconocimiento facial
│   └── utils.py                  # Utilidades varias
├── README.md                     # Este archivo
└── requirements.txt              # Archivo de requisitos

## Preparación de Datos

Para entrenar la red neuronal, necesitas un conjunto de datos de imágenes de rostros. Puedes usar datasets públicos o tus propios datos. Las imágenes deben estar organizadas en subdirectorios, cada uno representando una clase (persona). Ejemplo:

data/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...

## Entrenamiento del Modelo

Para entrenar el modelo de reconocimiento facial, ejecuta el script de entrenamiento que se encuentra en deep_neural_network.py:

from src.deep_neural_network import DeepNeuralNetwork, load_data

data_path = 'data/'
img_size = (64, 64)

X_train, Y_train, class_names = load_data(data_path, img_size)

layers_dims = [X_train.shape[0], 128, 64, len(class_names)]
nn = DeepNeuralNetwork(layers_dims)

nn.train(X_train, Y_train, learning_rate=0.01, epochs=1000)

# Guardar el modelo entrenado
import numpy as np
np.save('models/face_recognition_parameters.npy', nn.parameters)

## Uso del Modelo

Para utilizar el modelo entrenado en el reconocimiento facial, carga el modelo y usa la función de predicción:

from src.deep_neural_network import DeepNeuralNetwork, predict_face
import cv2

parameters = np.load('models/face_recognition_parameters.npy', allow_pickle=True).item()
nn = DeepNeuralNetwork(layers_dims)
nn.parameters = parameters

new_img_path = 'path_to_new_face_image'
new_img = cv2.imread(new_img_path)
new_img = cv2.resize(new_img, img_size).reshape(-1, 1) / 255.0

prediction = predict_face(nn, new_img)
predicted_class = class_names[prediction[0]]
print(f'Predicted class: {predicted_class}')

## Evaluación y Resultados

Para evaluar el rendimiento del modelo, se pueden usar métricas como la precisión, la exactitud y la pérdida en el conjunto de validación. Ejemplo:

test_loss, test_acc = model.evaluate(validation_generator, steps=50)
print('Test accuracy:', test_acc)

## Contribuciones

Las contribuciones no son bienvenidas

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

Autor: José Teodosio Lorente Vallecillos

Fecha: 12/08/2024

Universidad: Universidad de Granada

Tutor: Antonio Bailon
