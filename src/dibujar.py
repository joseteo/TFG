import numpy as np
import cv2
import matplotlib.pyplot as plt

def ImageProcessing():
    image = np.zeros((512,512,3), np.uint8)

    cv2.line(image, pt1=(20,200), pt2=(200,20), color=(0,0,255), thickness=5)
    cv2.rectangle(image, pt1=(200,60), pt2=(20,200), color=(255,0,0), thickness=3)
    cv2.circle(image, center=(80,80), radius=50, color=(0,255,0), thickness=4)

    mytext = "Hola Mundo"
    cv2.putText(image, mytext, (100,300), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255,255,255))

    vertices = np.array( [ [100,300],[300,200],[400,400],[200,400] ], np.int32)
    vertices.shape

    puntos = vertices.reshape(-1,1,2)
    puntos.shape

    cv2.polylines(image, [puntos], isClosed=True, color=(255,255,255), thickness=5)

    cv2.imshow("Black Image", image)
    cv2.waitKey(0) or " "
    cv2.destroyAllWindows()

ImageProcessing()