import logging
import cv2
import socket
import numpy as np
from kivy import platform
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.image import Image
import cv2.aruco as aruco

# Configuración básica del registro de eventos
logging.basicConfig(level=logging.DEBUG)

# Importación de permisos específicos para Android
if platform == "android":
    from android.permissions import request_permissions, Permission  # type: ignore

# Diccionario ArUco y parámetros
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()


# Clase principal del diseño de la aplicación
class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(MainLayout, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.video_feed = Image()
        self.add_widget(self.video_feed)
        self.status_label = Label(text="Initializing...", size_hint=(1, 0.1))
        self.add_widget(self.status_label)


# Clase principal de la aplicación
class ArSmartphone(App):
    def build(self):
        # Configuración inicial de la ventana
        Window.fullscreen = 'auto'
        Window.orientation = 'landscape'
        self.layout = MainLayout()

        # Verificación de permisos si la plataforma es Android
        if platform == "android":
            self.request_android_permissions()
        else:
            Clock.schedule_once(self.start_video_feed, 1)
        return self.layout

    # Solicitud de permisos para Android
    def request_android_permissions(self):
        def callback(permissions, results):
            if all(results):
                Clock.schedule_once(self.start_video_feed, 1)
            else:
                self.layout.status_label.text = "Permissions not granted. App cannot function."

        request_permissions([
            Permission.INTERNET,
            Permission.CAMERA,
            Permission.WRITE_EXTERNAL_STORAGE,
            Permission.READ_EXTERNAL_STORAGE
        ], callback)

    # Inicio de la transmisión de video y conexión al servidor
    def start_video_feed(self, dt):
        self.host = '192.168.68.102'  # Dirección IP del servidor
        self.send_port = 5000
        self.receive_port = 5001

        self.layout.status_label.text = "Initializing camera..."
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.layout.status_label.text = "Failed to open camera"
            return

        self.layout.status_label.text = "Connecting to server..."
        try:
            # Conexión con el servidor para enviar y recibir datos
            self.client_socket_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket_send.settimeout(10)
            self.client_socket_send.connect((self.host, self.send_port))
            self.connection_send = self.client_socket_send.makefile('wb')

            self.client_socket_receive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket_receive.settimeout(10)
            self.client_socket_receive.connect((self.host, self.receive_port))
            self.connection_receive = self.client_socket_receive.makefile('rb')
        except Exception as e:
            self.layout.status_label.text = f"Failed to connect: {str(e)}"
            self.cleanup()
            return

        self.layout.status_label.text = "Connected. Streaming..."
        self.running = True
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    # Limpieza de recursos al detener la aplicación o si ocurre un error
    def cleanup(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'client_socket_send'):
            self.client_socket_send.close()
        if hasattr(self, 'client_socket_receive'):
            self.client_socket_receive.close()

    # Detener la transmisión de video
    def stop_video_feed(self):
        self.running = False
        Clock.unschedule(self.update)
        self.cleanup()

    # Aplicación de distorsión de barril a la imagen
    def apply_barrel_distortion(self, image):
        # Parámetros de distorsión de barril
        K1 = -0.3  # Coeficiente de distorsión
        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2
        radius = np.sqrt(center_x ** 2 + center_y ** 2)

        # Mapa para la transformación
        map_x = np.zeros_like(image, dtype=np.float32)
        map_y = np.zeros_like(image, dtype=np.float32)

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                delta_x = x - center_x
                delta_y = y - center_y
                distance = np.sqrt(delta_x ** 2 + delta_y ** 2)
                r = distance / radius
                theta = 1 + K1 * (r ** 2)
                map_x[y, x] = center_x + theta * delta_x
                map_y[y, x] = center_y + theta * delta_y

        distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        return distorted_image

    # Detección de los marcadores y cálculo de la homografía
    def detect_and_calculate_homography(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) == 9:
            # Filtrar las esquinas de los 4 marcadores en las esquinas (IDs 1, 3, 7 y 9)
            corner_ids = [0, 2, 6, 8]  # IDs correspondientes a las esquinas
            corner_points = [corners[i] for i in corner_ids]

            # Coordenadas aproximadas en píxeles de destino para la homografía
            target_points = np.array([[0, 0], [800, 0], [800, 600], [0, 600]], dtype=np.float32)

            # Obtener las esquinas de los marcadores en el frame
            frame_corners = np.array([np.mean(corner[0], axis=0) for corner in corner_points], dtype=np.float32)

            # Calcular la homografía
            H, _ = cv2.findHomography(frame_corners, target_points)

            return H
        return None

    # Actualización del flujo de video y procesamiento de datos
    def update(self, dt):
        if not self.running:
            return

        try:
            ret, frame = self.cap.read()
            if not ret:
                self.layout.status_label.text = "Failed to capture frame"
                return

            # Envío del frame al servidor
            _, buffer = cv2.imencode('.jpg', frame)
            self.connection_send.write(len(buffer).to_bytes(4, byteorder='big'))
            self.connection_send.write(buffer.tobytes())
            self.connection_send.flush()

            # Recepción del frame procesado desde el servidor
            length_data = self.connection_receive.read(4)
            if not length_data:
                self.layout.status_label.text = "No data received for length"
                return

            length = int.from_bytes(length_data, byteorder='big')
            frame_data = self.connection_receive.read(length)
            if len(frame_data) != length:
                self.layout.status_label.text = f"Received incorrect data length"
                return

            processed_frame = np.frombuffer(frame_data, dtype=np.uint8)
            processed_frame = cv2.imdecode(processed_frame, cv2.IMREAD_COLOR)
            if processed_frame is None or processed_frame.size == 0:
                self.layout.status_label.text = "Frame decoding failed"
                return

            # Detección de marcadores y aplicación de homografía
            H = self.detect_and_calculate_homography(processed_frame)
            if H is not None:
                height, width, _ = frame.shape
                warped_frame = cv2.warpPerspective(processed_frame, H, (width, height))

                # Aplicar la distorsión de barril
                distorted_frame = self.apply_barrel_distortion(warped_frame)

                # Conversión a formato VR (lado a lado)
                vr_frame = np.concatenate((distorted_frame, distorted_frame), axis=1)

                # Conversión a textura para Kivy
                buf = cv2.flip(vr_frame, 0).tobytes()
                texture = Texture.create(size=(width * 2, height), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.layout.video_feed.texture = texture

        except Exception as e:
            self.layout.status_label.text = f"Error: {str(e)}"
            self.stop_video_feed()

    # Manejo de eventos al detener la aplicación
    def on_stop(self):
        self.stop_video_feed()


if __name__ == "__main__":
    ArSmartphone().run()
