import cv2
import numpy as np

dibujando = False
valorx = 0
valory = 0

#Funcion dibujar
def dibujar_rectangulo(event, x, y, etiquetas, parametros):
    global dibujando, valorx, valory #Globales estan definidas fuera
    if event == cv2.EVENT_LBUTTONDOWN:
        dibujando = True
        valorx = x
        valory = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if dibujando:
            cv2.rectangle(image, (valorx, valory), (x,y), (255, 0, 0), -1)

    if event == cv2.EVENT_LBUTTONUP:
        dibujando = False
        cv2.rectangle(image, (valorx, valory), (x, y), (255, 0, 0), -1) # el -1 indica que está relleno
        # elif forma == 'circle':
        #     cv2.circle(image, (x, y), 50, (255, 0, 0), -1)

cv2.namedWindow(winname= 'miImagen')
cv2.setMouseCallback('miImagen', dibujar_rectangulo())

image = np.zeros( (500,500,3), np.int8)

while True:
    cv2.imshow('miImagen', image)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cv2.destroyAllWindows()