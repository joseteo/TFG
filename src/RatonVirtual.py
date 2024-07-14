import cv2
import numpy as np
import SeguimientoManos as sm
from pynput.mouse import Controller, Button
import screeninfo

# Inicializar variables globales
teseracto = False

def process_hand(frame, anchocam=640, altocam=480, cuadro=100):
    # Inicializar variables
    sua = 5
    pubix, pubiy = 0, 0
    cubix, cubiy = 0, 0

    # Obtener las dimensiones de la pantalla
    screen = screeninfo.get_monitors()[0]
    anchopantalla, altopantalla = screen.width, screen.height

    detector = sm.detectormanos(maxManos=1)
    mouse = Controller()

    frame = detector.encontrarmanos(frame)
    lista, bbox = detector.encontrarposicion(frame)

    if len(lista) != 0:
        x1, y1 = lista[8][1:]
        x2, y2 = lista[12][1:]

        dedos = detector.dedosarriba()

        cv2.rectangle(frame, (cuadro, cuadro), (anchocam - cuadro, altocam - cuadro), (0, 0, 0), 2)

        if dedos[1] == 1 and dedos[2] == 0:
            x3 = np.interp(x1, (cuadro, anchocam - cuadro), (0, anchopantalla))
            y3 = np.interp(y1, (cuadro, altocam - cuadro), (0, altopantalla))

            cubix = pubix + (x3 - pubix) / sua
            cubiy = pubiy + (y3 - pubiy) / sua

            mouse.position = (anchopantalla - cubix, cubiy)
            cv2.circle(frame, (x1, y1), 10, (0, 0, 0), cv2.FILLED)
            pubix, pubiy = cubix, cubiy

        if dedos[1] == 1 and dedos[2] == 1:
            longitud, frame, linea = detector.distancia(8, 12, frame)
            if longitud < 30:
                cv2.circle(frame, (linea[4], linea[5]), 10, (0, 255, 0), cv2.FILLED)
                mouse.click(Button.left, 1)

        # Pintar el teseracto si se detectó la palabra "teseracto"
        global teseracto
        if teseracto:
            x, y, w, h = bbox
            frame = pintar_teseracto(frame, x, y, w, h)

    return frame

def pintar_teseracto(frame, x, y, w, h):
    # Dibujar un cubo azul del tamaño de la mano
    start_point = (x, y)
    end_point = (x + w, y + h)
    color = (255, 0, 0)  # Azul en BGR
    thickness = 2  # Grosor de las líneas

    # Dibujar el cubo
    frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
    return frame




# import cv2
# import numpy as np
# import SeguimientoManos as sm
# from pynput.mouse import Controller, Button
# import screeninfo
#
# anchocam, altocam = 640, 480
# cuadro = 100
#
# # Obtener las dimensiones de la pantalla
# screen = screeninfo.get_monitors()[0]
# anchopantalla, altopantalla = screen.width, screen.height
#
# sua = 5
# pubix, pubiy = 0, 0
# cubix, cubiy = 0, 0
#
# cap = cv2.VideoCapture(0)
# cap.set(3, anchocam)
# cap.set(4, altocam)
#
# detector = sm.detectormanos(maxManos=1)
# mouse = Controller()
#
# while True:
#     ret, frame = cap.read()
#     frame = detector.encontrarmanos(frame)
#     lista, bbox = detector.encontrarposicion(frame)
#
#     if len(lista) != 0:
#         x1, y1 = lista[8][1:]
#         x2, y2 = lista[12][1:]
#
#         dedos = detector.dedosarriba()
#
#         cv2.rectangle(frame, (cuadro, cuadro), (anchocam - cuadro, altocam - cuadro), (0, 0, 0), 2)
#
#         if dedos[1] == 1 and dedos[2] == 0:
#             x3 = np.interp(x1, (cuadro, anchocam - cuadro), (0, anchopantalla))
#             y3 = np.interp(y1, (cuadro, altocam - cuadro), (0, altopantalla))
#
#             cubix = pubix + (x3 - pubix) / sua
#             cubiy = pubiy + (y3 - pubiy) / sua
#
#             mouse.position = (anchopantalla - cubix, cubiy)
#             cv2.circle(frame, (x1, y1), 10, (0, 0, 0), cv2.FILLED)
#             pubix, pubiy = cubix, cubiy
#
#         if dedos[1] == 1 and dedos[2] == 1:
#             longitud, frame, linea = detector.distancia(8, 12, frame)
#             if longitud < 30:
#                 cv2.circle(frame, (linea[4], linea[5]), 10, (0, 255, 0), cv2.FILLED)
#                 mouse.click(Button.left, 1)
#
#     cv2.imshow("Raton", frame)
#     if cv2.waitKey(1) == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
