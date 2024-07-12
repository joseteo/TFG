import pygetwindow
import pyautogui
from PIL import Image
import platform

path = 'C:\\Users\\joset\\PycharmProjects\\pythonProject\\assets\\result.png'
titles = pygetwindow.getAllTitles()

# Aqui comprobamos si estamos en la arquitectura de systema Windows
if platform.system() == 'Windows':
    #window = pygetwindow.getAllWindows()[0]
    window = pygetwindow.getWindowsWithTitle(pygetwindow.getActiveWindowTitle()[0])[0]
    #window = pygetwindow.getwindowsWithTitle('Command Prompt')[0]
    #print(window)
    #print(window(0))
    left, top = window.topleft
    right, bottom = window.bottomright
    pyautogui.screenshot(path)
    im = Image.open(path)
    im = im.crop((left+10, top, right-10, bottom-10))
    im.save(path)
    im.show(path)
# O si estamos en la plataforma MasOS
elif platform.system() == 'Darwin':
    x1, y1, width, height = pygetwindow.getWindowGeometry('Terminal ')
    x2 = x1 + width
    y2 = y1 + height
    pyautogui.screenshot (path)
    im = Image.open(path)
    im= im.crop((x1, y1, x2, y2))
    im. save(path)
    im. show (path)