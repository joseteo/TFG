import cv2
import os
import numpy as np

#### URL A CAMBIAR ####
dataPath = 'src/FacialRecognition/Data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imagenes')

    for fileName in os.listdir(personPath):
        print('Rostro: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))     #con el 0 haces transformacion a escala de grises
        image = cv2.imread(personPath + '/' + fileName, 0)

    label = label + 1  
    '''print('Numero de etiquetas 0: ', np.count_nonzero(np.array(labels)==0))
    print('Numero de etiquetas 1: ', np.count_nonzero(np.array(labels)==1))
    test para comprobar que hay 300 0s y 300 1s, por lo tanto la asignacion de labels es correcta
    '''
face_recognizer = cv2.face.EigenFaceRecognizer_create()

#Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

#Almacenar el modelo obtenido
face_recognizer.write('modeloReconocimientoFacial.xml')
print("Modelo obternido almacenado")


cv2.destroyAllWindows()