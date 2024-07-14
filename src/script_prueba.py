import threading
import queue
import re
import pyttsx3
import speech_recognition as sr

# Crear cola para la comunicación
voice_queue = queue.Queue()

# Inicializar reconocimiento de voz
rec = sr.Recognizer()
mic = sr.Microphone()
engine = pyttsx3.init()
voices = engine.getProperty('voices')

# Variables globales
modo_r_d = False
circulo = False
rectangulo = False

def process_voice(voice_queue):
    global modo_r_d, circulo, rectangulo
    while True:
        result = voice_queue.get()
        person_name = 'Verita'
        try:
            with mic as source:
                rec.adjust_for_ambient_noise(source, duration=0.5)
                audio = rec.listen(source)

                texto = rec.recognize_google(audio, language='es-ES')

                if re.search(r"Alfred", texto, re.IGNORECASE):
                    print(f'\n Hola {person_name}')

                    engine.setProperty('rate', 160)
                    engine.setProperty('voice',
                                       'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ES-ES_HELENA_11.0')
                    engine.say(f'Hola {person_name}')
                    engine.runAndWait()

                    if re.search(r"Ratón|ratón|Raton|raton", texto, re.IGNORECASE):
                        print('Seleccionada Mano Raton Virtual')
                        modo_r_d = False
                    elif re.search(r"Dibujar|circulo", texto, re.IGNORECASE) and (result[0] == 0 or result[0] == 2):
                        modo_r_d = True
                        circulo = True
                        rectangulo = False
                        print('Ha seleccionado dibujar la forma circulo')
                    elif re.search(r"Dibujar|rectangle", texto, re.IGNORECASE) and (result[0] == 0 or result[0] == 2):
                        modo_r_d = True
                        rectangulo = True
                        circulo = False
                        print('Ha seleccionado dibujar la forma rectangulo')

        except sr.RequestError:
            print("API no disponible")
        except sr.UnknownValueError:
            print("")

# Función para agregar datos a la cola y probar
def add_to_queue(voice_queue, result, person_name):
    voice_queue.put((result, person_name))

# Iniciar el hilo de procesamiento de voz
voice_thread = threading.Thread(target=process_voice, args=(voice_queue,))
voice_thread.start()

# Agregar un ejemplo de prueba a la cola
add_to_queue(voice_queue, (0, 'PersonName'), "Alfred")

# Esperar a que se procese el ejemplo
voice_thread.join()
