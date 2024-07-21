# from vidstream import ScreenShareClient
# import threading
#
# sender = ScreenShareClient('192.168.56.1', 9999)
#
# t = threading.Thread(target=sender.start_stream())
# t.start()
#
# while input("") != 'STOP':
#     continue
#
# sender.stop_stream()

from vidstream import ScreenShareClient
import threading
import pygetwindow as gw
import platform
from CaptureWindow import capture_and_stream_window

sender = None
streaming_thread = None


def start_streaming():
    global sender, streaming_thread
    if platform.system() == 'Windows':
        titles = gw.getAllTitles()
        active_window_title = gw.getActiveWindow().title
        print(f"Capturing window: {active_window_title}")

        sender = ScreenShareClient('192.168.56.1', 9999)

        streaming_thread = threading.Thread(target=capture_and_stream_window, args=(active_window_title, sender))
        streaming_thread.start()
    elif platform.system() == 'Darwin':
        print("This functionality is not implemented for macOS in this script.")


def stop_streaming():
    global sender, streaming_thread
    if sender:
        sender.stop_stream()
    if streaming_thread:
        streaming_thread.join()


start_streaming()

while input("") != 'STOP':
    continue

stop_streaming()
