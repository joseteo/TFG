import pyttsx3
import speech_recognition as sr
import pyaudio
from gtts import gTTS
import IPython.display as ipd
import re

rec = sr.Recognizer()
mic = sr.Microphone()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
#for voice in voices:
    #print(voice)
#tts = gTTS(text="Texto de prueba", lang='es')
#tts.save("audio.mp3")
#ipd.Audio("audio.mp3")

while True:
    try:
        with mic as source:
            rec.adjust_for_ambient_noise(source, duration=0.5)
            audio = rec.listen(source)

            texto = rec.recogniz
            e_google(audio, language='es-ES')
            t_aux = "Hola Teo, que puedo hacer por ti"
            if re.search(r"Alfred", texto, re.IGNORECASE):
                print(t_aux)
                engine.say(t_aux)

            engine.setProperty('rate', 160)
            engine.setProperty('voice',
                               'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-ES_HELENA_11.0')
            engine.say(texto);
            engine.save_to_file(texto, 'audio.mp3')
            engine.runAndWait()
            ipd.Audio("audio.mp3")
            print(f'Has dicho: {texto}')
    except sr.RequestError:
        print("API no disponible")
    except sr.UnknownValueError:
        print("No se pudo reconocer el audio")