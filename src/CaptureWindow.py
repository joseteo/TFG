import pygetwindow as gw
import pyautogui
import cv2
import numpy as np
import mss
import os
from PIL import Image


def load_mouse_cursor_image(png_file, width, height):
    # Load the PNG file as an image
    image = Image.open(png_file).convert("RGBA")
    # Resize the image
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

    alpha_overlay = overlay[y1o:y2o, x1o:x2o, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(3):
        background[y1:y2, x1:x2, c] = (alpha_overlay * overlay[y1o:y2o, x1o:x2o, c] +
                                       alpha_background * background[y1:y2, x1o:x2o, c])

    return background


def capture_and_stream_window(window_title, sender):
    # Find the window with the given title
    window = gw.getWindowsWithTitle(window_title)[0]
    left, top = window.topleft
    right, bottom = window.bottomright

    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'cursor-icon.png')
    mouse_cursor_img = load_mouse_cursor_image(assets_dir, 15, 20)

    with mss.mss() as sct:
        while True:
            # Define the area to capture
            monitor = {"top": top, "left": left, "width": right - left, "height": bottom - top}
            # Capture the screen
            sct_img = sct.grab(monitor)
            # Convert the image to a format that OpenCV can use
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Get the mouse position
            mouse_x, mouse_y = pyautogui.position()

            # Overlay the mouse cursor on the image
            if left <= mouse_x <= right and top <= mouse_y <= bottom:
                img = overlay_image(img, mouse_cursor_img, mouse_x - left, mouse_y - top)

            # Resize the image to improve performance
            img = cv2.resize(img, (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)

            # Stream the image
            _, buffer = cv2.imencode('.jpg', img)
            sender.send_frame(buffer.tobytes())

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


# if __name__ == "__main__":
#     if platform.system() == 'Windows':
#         titles = gw.getAllTitles()
#         active_window_title = gw.getActiveWindow().title
#         print(f"Capturing window: {active_window_title}")
#
#         sender = ScreenShareClient('192.168.56.1', 9999)
#         sender.start_stream()
#         capture_and_stream_window(active_window_title, sender)
#     elif platform.system() == 'Darwin':
#         print("This functionality is not implemented for macOS in this script.")

# def capture_window(window_title):
#     # Find the window with the given title
#     window = gw.getWindowsWithTitle(window_title)[0]
#     left, top = window.topleft
#     right, bottom = window.bottomright
#
#     assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'cursor-icon.png')
#     mouse_cursor_img = load_mouse_cursor_image(assets_dir, 15, 20)
#
#     cv2.namedWindow(window_title, cv2.WND_PROP_FULLSCREEN)
#     cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#
#     with mss.mss() as sct:
#         while True:
#             # Define the area to capture
#             monitor = {"top": top, "left": left, "width": right - left, "height": bottom - top}
#             # Capture the screen
#             sct_img = sct.grab(monitor)
#             # Convert the image to a format that OpenCV can use
#             img = np.array(sct_img)
#             img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
#
#             # Get the mouse position
#             mouse_x, mouse_y = pyautogui.position()
#
#             # Overlay the mouse cursor on the image
#             if left <= mouse_x <= right and top <= mouse_y <= bottom:
#                 img = overlay_image(img, mouse_cursor_img, mouse_x - left, mouse_y - top)
#
#             # Resize the image to improve performance
#             img = cv2.resize(img, (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)
#
#             # Display the image in a window
#             cv2.imshow(window_title, img)
#
#             # Exit the loop if the 'q' key is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#     # Close the window
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     if platform.system() == 'Windows':
#         titles = gw.getAllTitles()
#         active_window_title = gw.getActiveWindow().title
#         print(f"Capturing window: {active_window_title}")
#         capture_window(active_window_title)
#     elif platform.system() == 'Darwin':
#         print("This functionality is not implemented for macOS in this script.")
