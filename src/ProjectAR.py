import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

mp_dibujo = mp.solutions.drawing_utils
mp_manos = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# Configurar la transmisión en vivo
cap = cv2.VideoCapture(0)
cv2.face.LBPHFaceRecognizer_create()
with mp_manos.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as manos:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se puede abrir la camara")
            break

        # Convertir el frame a RGB para procesarla con MediaPipe
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        # Detectar las manos en el frame
        results = manos.process(frame)

        # Dibujar las marcas y la caja delimitadora alrededor de las manos
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_dibujo.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_manos.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        cv2.imshow('Detector Manos', frame)

        # Salir del programa si se presiona ' '
        if cv2.waitKey(1) == ord(" "):
            break

# Limpiar la memoria
cap.release()
cv2.destroyAllWindows()



#*************************************************
#*************************************************
# Hacer coincidir marcador con los puntos de las manos:
#   - 4: Punta del Pulgar
#   - 8: Punta del Indice
#   - 12: Punta del Corazon
#   - 16: Punta del Anular
#   - 20: Putna del Meñique

#O. WRIST

#1. THUMB-CMC
#2. THUMB-MCP
#3. THUMB-IP
#4. THUMB-TIP

#5. INDEX-FINGER-MCP
#6. INDEX-FINGER-PIP
#7. INDEX-FINGER-DIP
#8. INDEX-FINGER-TIP

#9.  MIDDLE-FINGER-MCP
#10. MIDDLE-FINGER-PIP
#11. MIDDLE-FINGER-DIP
#12. MIDDLE-FINGER-TIP

#13. RING-FINGER-MCP
#14. RING-FINGER-PIP
#15. RING-FINGER-DIP
#16. RING-FINGER-TIP

#17. PINKY-MCP
#18. PINKY-PIP
#19. PINKY-DIP
#20. PINKY-TIP

# Basar las cuatro esquinas del marcador/objeto con en el PULGAR, MEÑIQUE, INDICE Y (CORAZON ó ANULAR)


#1. Primero, debes obtener las coordenadas 3D de los puntos clave de la mano detectados por MediaPipe Hands.
# Puedes hacer esto utilizando la función hand_landmarks.landmark, que devuelve la posición de cada punto clave en la imagen en términos de píxeles. 
# Luego, puedes utilizar la función hand_landmarks.landmark[i].x y hand_landmarks.landmark[i].y para obtener las coordenadas x e y del punto clave en la imagen.

#2. Una vez que tengas las coordenadas 2D de los puntos clave, debes proyectarlas en el espacio 3D.
# Puedes hacer esto utilizando la función cv2.projectPoints de OpenCV, que toma las coordenadas 3D de un objeto y las proyecta en la imagen.

#3. Ahora, tienes las coordenadas 3D de los puntos clave de la mano. 
# Puedes utilizar estas coordenadas para crear un objeto 3D en Matplotlib. 
# Por ejemplo, puedes crear un cubo utilizando la función mpl_toolkits.mplot3d.art3d.Poly3DCollection, que toma una lista de vértices en 3D y las caras del cubo.






# PARA LA MEMORIA, PONER QUE SE HA TENIDO EN CUENTA LA DETECCION DE LAS MANOS DE PERSONAS CON DIFERENTES TONOS DE PIEL

#*************************************************
#*************************************************