import os
from socket import socket
from threading import Thread
from zlib import compress

import cv2
from mss import mss
import pyautogui
import numpy as np
from PIL import Image

WIDTH = 1900
HEIGHT = 1000
CURSOR_SIZE = 20


def load_mouse_cursor_image(path, width, height):
    image = Image.open(path).convert("RGBA")
    image = image.resize((width, height), Image.LANCZOS)
    return np.array(image)


def overlay_image(background, overlay, x, y):
    h, w, _ = overlay.shape
    rows, cols, _ = background.shape

    if x >= cols or y >= rows:
        return background

    y1, y2 = max(0, y), min(rows, y + h)
    x1, x2 = max(0, x), min(cols, x + w)

    y1o, y2o = max(0, -y), min(h, rows - y)
    x1o, x2o = max(0, -x), min(w, cols - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return background

    alpha_overlay = overlay[y1o:y2o, x1o:x2o, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(3):
        background[y1:y2, x1:x2, c] = (alpha_overlay * overlay[y1o:y2o, x1o:x2o, c] +
                                       alpha_background * background[y1:y2, x1o:x2o, c])

    return background


def retrieve_screenshot(conn):
    with mss() as sct:
        rect = {'top': 0, 'left': 0, 'width': WIDTH, 'height': HEIGHT}
        cursor_img = load_mouse_cursor_image(os.path.join('..', 'assets', 'cursor-icon.png'), CURSOR_SIZE, CURSOR_SIZE + 6)

        while True:
            img = np.array(sct.grab(rect))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            mouse_x, mouse_y = pyautogui.position()
            img = overlay_image(img, cursor_img, mouse_x, mouse_y)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Compress the image
            img = img[:, :, :3]
            pixels = compress(img.tobytes(), 6)

            # Send the size of the pixels length
            size = len(pixels)
            size_len = (size.bit_length() + 7) // 8
            conn.send(bytes([size_len]))

            # Send the actual pixels length
            size_bytes = size.to_bytes(size_len, 'big')
            conn.send(size_bytes)

            # Send pixels
            conn.sendall(pixels)


def main(host='0.0.0.0', port=5000):
    sock = socket()
    sock.bind((host, port))
    try:
        sock.listen(5)
        print('Server started.')

        while True:
            conn, addr = sock.accept()
            print('Client connected IP:', addr)
            thread = Thread(target=retrieve_screenshot, args=(conn,))
            thread.start()
    finally:
        sock.close()


if __name__ == '__main__':
    main()
