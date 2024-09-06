import socket
import cv2
import numpy as np
import cv2.aruco as aruco
import mss
from CaptureWindow import capture_window_continuous, overlay_image


def calculate_center_of_markers(aruco_corners):
    """
    Calcula el centro de los marcadores visibles utilizando el promedio de sus coordenadas.
    """
    if len(aruco_corners) == 0:
        return None

    all_corners = np.concatenate(aruco_corners, axis=1)
    center_x = np.mean(all_corners[0, :, 0])
    center_y = np.mean(all_corners[0, :, 1])

    return int(center_x), int(center_y)


def calculate_average_marker_width(aruco_corners):
    """
    Calcula el ancho promedio de los marcadores visibles.
    """
    widths = []
    for corners in aruco_corners:
        width = np.linalg.norm(corners[0][0] - corners[0][1])  # Calcula el ancho del marcador
        widths.append(width)

    return np.mean(widths)


def process_and_merge_frames(client_frame, pc_frame, aruco_corners):
    # Calcular el centro de los marcadores visibles
    center_of_markers = calculate_center_of_markers(aruco_corners)
    if center_of_markers is None:
        return client_frame  # Si no hay marcadores visibles, devolver el frame del cliente sin cambios

    # Calcular el ancho promedio de los marcadores visibles
    avg_marker_width = calculate_average_marker_width(aruco_corners)

    # Redimensionar la captura de pantalla del PC al 80% del ancho promedio de los marcadores
    scale_factor = avg_marker_width * 0.8 / pc_frame.shape[1]
    new_width = int(pc_frame.shape[1] * scale_factor)
    new_height = int(pc_frame.shape[0] * scale_factor)
    resized_pc_frame = cv2.resize(pc_frame, (new_width, new_height))

    # Posicionar la captura de pantalla en el centro de la figura formada por los marcadores
    top_left_x = center_of_markers[0] - resized_pc_frame.shape[1] // 2
    top_left_y = center_of_markers[1] - resized_pc_frame.shape[0] // 2

    # Superponer la captura de pantalla sobre el frame del cliente
    output_frame = client_frame.copy()
    output_frame = overlay_image(output_frame, resized_pc_frame, top_left_x, top_left_y)

    return output_frame


def handle_client_connection(connection, window_title):
    try:
        # Captura continua del escritorio del PC
        pc_frames = capture_window_continuous(window_title)

        while True:
            # Recibir frame del cliente (móvil)
            client_frame_data = connection.recv(921600)  # Tamaño típico de un frame 640x480 en JPEG
            if not client_frame_data:
                break
            client_frame = cv2.imdecode(np.frombuffer(client_frame_data, np.uint8), cv2.IMREAD_COLOR)

            # Procesar los marcadores ArUco aquí para obtener las esquinas (aruco_corners)
            aruco_corners, ids, _ = cv2.aruco.detectMarkers(client_frame, aruco_dict, parameters=aruco_params)

            if len(aruco_corners) > 0:
                # Obtener la captura de pantalla del PC
                pc_frame = next(pc_frames)

                # Fusionar los frames
                merged_frame = process_and_merge_frames(client_frame, pc_frame, aruco_corners)

                # Enviar el frame fusionado de vuelta al cliente
                _, buffer = cv2.imencode('.jpg', merged_frame)
                connection.sendall(buffer.tobytes())

    finally:
        connection.close()


def run_server():
    # Configuración del servidor
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 5000))
    server_socket.listen(1)
    print("Esperando conexión...")
    connection, address = server_socket.accept()
    print(f"Conectado a {address}")

    # Obtener el título de la ventana activa
    active_window_title = gw.getActiveWindow().title

    # Manejar la conexión con el cliente
    handle_client_connection(connection, active_window_title)

    server_socket.close()


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

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
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
