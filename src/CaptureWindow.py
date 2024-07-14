import pygetwindow as gw
import pyautogui
import cv2
import numpy as np
import mss
import platform


def capture_window(window_title):
    # Encuentra la ventana con el título dado
    window = gw.getWindowsWithTitle(window_title)[0]
    left, top = window.topleft
    right, bottom = window.bottomright

    with mss.mss() as sct:
        while True:
            # Define el área a capturar
            monitor = {"top": top, "left": left, "width": right - left, "height": bottom - top}
            # Captura la pantalla
            sct_img = sct.grab(monitor)
            # Convierte la imagen a un formato que OpenCV pueda usar
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Muestra la imagen en una ventana
            cv2.imshow(window_title, img)

            # Salir del bucle si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cierra la ventana
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if platform.system() == 'Windows':
        titles = gw.getAllTitles()
        active_window_title = gw.getActiveWindow().title
        print(f"Capturando la ventana: {active_window_title}")
        capture_window(active_window_title)
    elif platform.system() == 'Darwin':
        print("Esta funcionalidad no está implementada para macOS en este script.")
