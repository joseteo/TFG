import socket
import cv2


def send_frames(host, port):
    cap = cv2.VideoCapture(0)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    connection = client_socket.makefile('wb')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            connection.write(buffer.tobytes())
            connection.flush()
    finally:
        cap.release()
        client_socket.close()


if __name__ == "__main__":
    send_frames('192.168.68.102', 5000)
