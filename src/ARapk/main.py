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

logging.basicConfig(level=logging.DEBUG)

if platform == "android":
    from android.permissions import request_permissions, Permission  # type: ignore

class MainLayout(BoxLayout):
    def _init_(self, **kwargs):
        super(MainLayout, self)._init_(**kwargs)
        self.orientation = 'vertical'
        self.video_feed = Image()
        self.add_widget(self.video_feed)
        self.status_label = Label(text="Initializing...", size_hint=(1, 0.1))
        self.add_widget(self.status_label)

class ArSmartphone(App):
    def build(self):
        Window.fullscreen = 'auto'
        Window.orientation = 'landscape'
        self.layout = MainLayout()
        if platform == "android":
            self.request_android_permissions()
        else:
            Clock.schedule_once(self.start_video_feed, 1)
        return self.layout
    
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

    def start_video_feed(self, dt):
        self.host = '192.168.68.102'  # Ensure this is the correct IP
        self.send_port = 5000
        self.receive_port = 5001

        self.layout.status_label.text = "Initializing camera..."
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.layout.status_label.text = "Failed to open camera"
            return

        self.layout.status_label.text = "Connecting to server..."
        try:
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

    def cleanup(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'client_socket_send'):
            self.client_socket_send.close()
        if hasattr(self, 'client_socket_receive'):
            self.client_socket_receive.close()

    def stop_video_feed(self):
        self.running = False
        Clock.unschedule(self.update)
        self.cleanup()

    def update(self, dt):
        if not self.running:
            return

        try:
            ret, frame = self.cap.read()
            if not ret:
                self.layout.status_label.text = "Failed to capture frame"
                return

            _, buffer = cv2.imencode('.jpg', frame)
            self.connection_send.write(len(buffer).to_bytes(4, byteorder='big'))
            self.connection_send.write(buffer.tobytes())
            self.connection_send.flush()

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

            # Convert to VR format (side-by-side)
            height, width, _ = processed_frame.shape
            vr_frame = np.concatenate((processed_frame, processed_frame), axis=1)

            # Convert to texture
            buf = cv2.flip(vr_frame, 0).tobytes()
            texture = Texture.create(size=(width * 2, height), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.layout.video_feed.texture = texture

        except Exception as e:
            self.layout.status_label.text = f"Error: {str(e)}"
            self.stop_video_feed()

    def on_stop(self):
        self.stop_video_feed()

if __name__ == "_main_":
    ArSmartphone().run()