import socket
import cv2
import numpy as np
import cv2.aruco as aruco
import mss


def receive_frames(host, send_port, receive_port):
    server_socket_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket_send.bind((host, send_port))
    server_socket_send.listen(1)
    connection_send, client_address_send = server_socket_send.accept()
    connection_send = connection_send.makefile('rb')

    server_socket_receive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket_receive.bind((host, receive_port))
    server_socket_receive.listen(1)
    connection_receive, client_address_receive = server_socket_receive.accept()
    connection_receive = connection_receive.makefile('wb')

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()

    with mss.mss() as sct:
        while True:
            try:
                # Read the length of the incoming message
                length = int.from_bytes(connection_send.read(4), byteorder='big')
                if length == 0:
                    continue

                # Read the frame data based on the length
                frame_data = connection_send.read(length)
                if len(frame_data) != length:
                    print(f"Expected {length} bytes, received {len(frame_data)} bytes")
                    continue

                # Convert frame data to numpy array
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

                if frame is None:
                    print("Frame decoding failed")
                    continue

                # frame = frame.reshape((480, 640, 3))

                # Detect ArUco markers
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

                # Capture PC screen
                monitor = {"top": 0, "left": 0, "width": 800, "height": 600}
                sct_img = sct.grab(monitor)
                pc_frame = np.array(sct_img)
                pc_frame = cv2.cvtColor(pc_frame, cv2.COLOR_BGRA2BGR)

                # Overlay PC screen capture within the area bounded by ArUco markers
                if ids is not None and len(ids) == 9:
                    # Perform transformation and overlay screen capture
                    pass

                # Send processed frame back to the smartphone
                _, buffer = cv2.imencode('.jpg', frame)
                connection_receive.write(len(buffer).to_bytes(4, byteorder='big'))
                connection_receive.write(buffer.tobytes())
                connection_receive.flush()

                # Display the received frame
                cv2.imshow('Server Received Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error: {e}")

    connection_send.close()
    connection_receive.close()
    server_socket_send.close()
    server_socket_receive.close()


if __name__ == "__main__":
    receive_frames('0.0.0.0', 5000, 5001)
