# Augmented Reality Project with Deep Neural Network

## Description

This project implements an augmented reality system using deep neural networks (DNNs) for facial and voice recognition. The goal is to create a robust tool capable of identifying individuals from facial images and recognizing voices using advanced image processing and machine learning techniques. This project aims to integrate these capabilities into a virtual reality setup with a low budget.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Project Structure](#project-structure)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Using the Model](#using-the-model)
7. [Evaluation and Results](#evaluation-and-results)
8. [Contributions](#contributions)
9. [License](#license)

## Introduction

This project is part of my Bachelor's Thesis and is designed to demonstrate the use of deep neural networks in augmented reality applications, specifically for facial and voice recognition. The system is composed of several modules, including image capture, face detection, image processing, and recognition using DNNs. The ultimate goal is to integrate this system into a virtual reality headset with added cameras for augmented reality capabilities.

## Requirements

To run this project, you need to install the following dependencies:

- Python 3.7+
- NumPy
- OpenCV
- TensorFlow / Keras
- PyAudio
- Pyttsx3
- Pynput
- Screeninfo

You can install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Project Structure

The project is organized as follows:
```
TFG/
├── assets/                          # Project resources
│   └── result.png                   # Result image
├── autopy/                          # Autopy resources
├── myenv/                           # Python virtual environment
├── src/                             # Source code of the project
│   ├── FacialRecognition/           # Facial recognition module
│   │   ├── Data/                    # Data directory for facial recognition
│   │   └── models/                  # Directory to save trained models
│   │       ├── class_names.npy
│   │       ├── face_recognition_model.h5
│   │       └── modeloReconocimientoFacial.xml
│   ├── VoiceRecognition/            # Voice recognition module
│   │   ├── assets/                  # Voice recognition assets
│   │   │   ├── confusion_matrix.png
│   │   │   ├── roc_curve.png
│   │   │   └── training_history.png
│   │   └── models/                  # Directory to save trained models
│   │       ├── voice_recognition_model.h5
│   │       └── voice_recognition_model.keras
│   ├── CapturaPantallaAplicacionAbierta.py # Screen capture script
│   ├── CaptureWindow.py             # Window capture script
│   ├── CustomScaleSlayerDNN.py      # Custom scale layer DNN implementation
│   ├── DeepNeuralNetwork.py         # Deep neural network implementation
│   ├── Draw.py                      # Drawing functions
│   ├── DrawMouse.py                 # Script to draw with the mouse
│   ├── DrawMoveMouse.py             # Script to draw by moving the mouse
│   ├── FaceRecognizer.py            # Script to recognize faces
│   ├── FacialRecognition.py         # Face recognition functions
│   ├── FacialRecognition_DNN.py     # Face recognition using DNN
│   ├── HandsTracking.py             # Hand tracking
│   ├── main.py                      # Main project file
│   ├── PrintDNN.py                  # Script to print DNN structure
│   ├── ProjectAR.py                 # Augmented reality project script
│   ├── ProjectAR_test.py            # Augmented reality project test script
│   ├── TrainingFR.py                # Script to train face recognition
│   ├── TrainingFR_DNN.py            # Script to train face recognition using DNN
│   ├── TrainingFR_DNN_Inception-ResNet.py # Training face recognition with Inception-ResNet
│   ├── VirtualMouse.py              # Script to control the virtual mouse
│   ├── VoiceRecognition.py          # Voice recognition functions
│   ├── VoiceRecognition_CNN.py      # Voice recognition using CNN
│   └── VoiceRecognition_testing.py  # Voice recognition testing script
├── .cache/                          # Cache directory
├── .gitattributes                   # Git attributes
├── audio.mp3                        # Audio file
├── README.md                        # This file
├── LICENSE                          # License file
├── requirements.txt                 # Requirements file
└── yolov5s.pt                       # Pre-trained YOLOv5 model
```
## Data Preparation

For training the neural network, you need datasets of facial images and voice samples. The images and voice samples should be organized into subdirectories, each representing a class (person). Example:

### Facial Recognition Data
```
FacialRecognition/Data/
├── user1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── user2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── user3/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Voice Recognition Data
```
VoiceRecognition/Data/
├── user1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── user2/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── user3/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

## Model Training

### Facial Recognition Model Training

To train the facial recognition model, run the training script located in TrainingFR_DNN.py:

```python
from DeepNeuralNetwork import DeepNeuralNetwork, load_data

data_path = 'FacialRecognition/Data/'
img_size = (64, 64)

X_train, Y_train, class_names = load_data(data_path, img_size)

layers_dims = [X_train.shape[1], 128, 64, len(class_names)]
nn = DeepNeuralNetwork(layers_dims)

nn.train(X_train, Y_train, learning_rate=0.01, epochs=1000)
```

### Saving the Trained Model

```python
import numpy as np
nn.save_model('FacialRecognition/models/face_recognition_model.h5')
np.save('FacialRecognition/models/class_names.npy', class_names)
```

### Voice Recognition Model Training

To train the voice recognition model, run the training script located in VoiceRecognition_CNN.py.

## Using the Model

### Facial Recognition

To use the trained facial recognition model:

```python
from DeepNeuralNetwork import DeepNeuralNetwork, predict_face
import numpy as np
import cv2

class_names = np.load('FacialRecognition/models/class_names.npy', allow_pickle=True)
nn = DeepNeuralNetwork.load_model('FacialRecognition/models/face_recognition_model.h5')

new_img_path = 'path_to_new_face_image'
new_img = preprocess_image(new_img_path)
prediction = predict_face(nn, new_img)
predicted_class = class_names[prediction[0]]
print(f'Predicted class: {predicted_class}')
```

### Voice Recognition

To use the trained voice recognition model, follow a similar procedure to load the model and make predictions.

## Evaluation and Results

To evaluate the model's performance, you can use metrics such as accuracy and loss on the validation set. Example:
```python
test_loss, test_acc = model.evaluate(validation_generator, steps=50)
print('Test accuracy:', test_acc)
```

## Contributions

Contributions are not welcome.

## License

This project is protected under a Proprietary License. See the LICENSE file for more details.

Author: José Teodosio Lorente Vallecillos

Date: 12/08/2024

University: University of Granada

Tutor: Antonio Bailon
