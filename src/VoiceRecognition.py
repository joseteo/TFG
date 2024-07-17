import pyttsx3
import speech_recognition as sr
import re

# Inicializar el reconocimiento de voz y el motor de texto a voz
rec = sr.Recognizer()
mic = sr.Microphone()
engine = pyttsx3.init()

# Configurar el motor de texto a voz
voices = engine.getProperty('voices')
engine.setProperty('voice', r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-ES_HELENA_11.0')

# Variables globales
modo_r_d, circulo, rectangulo, teseracto = False, False, False, False

# Función para procesar el reconocimiento de voz
def process_voice(voice_queue):
    global modo_r_d, circulo, rectangulo, teseracto
    while True:
        result, person_name = voice_queue.get()
        try:
            with mic as source:
                rec.adjust_for_ambient_noise(source, duration=0.5)
                audio = rec.listen(source)

                texto = rec.recognize_google(audio, language='es-ES')

                if re.search(r"Alfred", texto, re.IGNORECASE):
                    print(f'\n Hola {person_name}')

                    engine.setProperty('rate', 160)
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
                    elif re.search(r"teseracto", texto, re.IGNORECASE):
                        teseracto = True
                        print('Ha seleccionado dibujar un teseracto')

        except sr.RequestError:
            print("API no disponible")
        except sr.UnknownValueError:
            print("")
