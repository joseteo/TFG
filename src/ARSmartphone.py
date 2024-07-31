import socket
import cv2
import numpy as np


def send_frames(host, send_port, receive_port):
    cap = cv2.VideoCapture(0)

    client_socket_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket_send.connect((host, send_port))
    connection_send = client_socket_send.makefile('wb')

    client_socket_receive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket_receive.connect((host, receive_port))
    connection_receive = client_socket_receive.makefile('rb')

    try:
        while True:
            # Capture frame from camera
            ret, frame = cap.read()
            if not ret:
                break

            # Send frame to server
            _, buffer = cv2.imencode('.jpg', frame)
            connection_send.write(len(buffer).to_bytes(4, byteorder='big'))
            connection_send.write(buffer.tobytes())
            connection_send.flush()

            # Receive processed frame from server
            length = int.from_bytes(connection_receive.read(4), byteorder='big')
            frame_data = connection_receive.read(length)
            if len(frame_data) != length:
                print(f"Expected {length} bytes, received {len(frame_data)} bytes")
                continue

            processed_frame = np.frombuffer(frame_data, dtype=np.uint8)
            processed_frame = cv2.imdecode(processed_frame, cv2.IMREAD_COLOR)

            if processed_frame is None:
                print("Processed frame decoding failed")
                continue

            # Display processed frame
            cv2.imshow('Processed Frame', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        client_socket_send.close()
        client_socket_receive.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    send_frames('192.168.68.102', 5000, 5001)
