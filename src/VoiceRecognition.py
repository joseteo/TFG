from VoiceRecognition_LSTM import VoiceRecognitionCNN
import speech_recognition as sr

class VoiceRecognitionSystem:
    def __init__(self, username):
        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()
        self.voice_recognition = VoiceRecognitionCNN()
        self.voice_recognition.initialize_model()
        self.username = username

    def capture_audio(self):
        """
        Captures audio from the microphone and returns the audio data.

        Returns:
            audio_data (sr.AudioData): The recorded audio data.
        """
        print("Listening...")
        with self.mic as source:
            self.rec.adjust_for_ambient_noise(source, duration=0.5)  # Adjust for ambient noise
            audio_data = self.rec.listen(source)  # Capture the audio from the microphone
        return audio_data

    def process_voice(self, voice_queue):
        """
        Processes voice commands for the recognized user.

        Args:
            voice_queue (Queue): A queue to get the recognized voice command.
        """
        while True:
            result, person_name = voice_queue.get()

            # Capture the audio command
            audio_data = self.capture_audio()

            # Train the voice model on the new audio data for the recognized user
            self.voice_recognition.train_on_new_data(audio_data=audio_data, user_label=self.username)

            # Recognize the command from the audio data
            recognized_command = self.voice_recognition.recognize_command(audio_data=audio_data)
            print(f'Recognized Command: {recognized_command}')

            # Perform actions based on the recognized command
            if recognized_command == "some_command":
                # Perform the action for the recognized command
                pass

            # If command is to log out the user, close the session
            if "cerrar sesion" in recognized_command.lower() and self.username.lower() in recognized_command.lower():
                print(f'Cerrando sesión para {self.username}')
                break


# Example usage
# process_voice(voice_queue, username="recognized_username")


# import pyttsx3
# import speech_recognition as sr
# import re

# # Inicializar el reconocimiento de voz y el motor de texto a voz
# rec = sr.Recognizer()
# mic = sr.Microphone()
# engine = pyttsx3.init()

# # Configurar el motor de texto a voz
# voices = engine.getProperty('voices')
# engine.setProperty('voice', r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-ES_HELENA_11.0')

# # Variables globales
# modo_r_d, circulo, rectangulo, teseracto = False, False, False, False

# # Función para procesar el reconocimiento de voz
# def process_voice(voice_queue):
#     global modo_r_d, circulo, rectangulo, teseracto
#     while True:
#         result, person_name = voice_queue.get()
#         try:
#             with mic as source:
#                 rec.adjust_for_ambient_noise(source, duration=0.5)
#                 audio = rec.listen(source)

#                 texto = rec.recognize_google(audio, language='es-ES')

#                 if re.search(r"Alfred", texto, re.IGNORECASE):
#                     print(f'\n Hola {person_name}')

#                     engine.setProperty('rate', 160)
#                     engine.say(f'Hola {person_name}')
#                     engine.runAndWait()

#                     if re.search(r"Ratón|ratón|Raton|raton", texto, re.IGNORECASE):
#                         print('Seleccionada Mano Raton Virtual')
#                         modo_r_d = False
#                     elif re.search(r"Dibujar|circulo", texto, re.IGNORECASE) and (result[0] == 0 or result[0] == 2):
#                         modo_r_d = True
#                         circulo = True
#                         rectangulo = False
#                         print('Ha seleccionado dibujar la forma circulo')
#                     elif re.search(r"Dibujar|rectangle", texto, re.IGNORECASE) and (result[0] == 0 or result[0] == 2):
#                         modo_r_d = True
#                         rectangulo = True
#                         circulo = False
#                         print('Ha seleccionado dibujar la forma rectangulo')
#                     elif re.search(r"teseracto", texto, re.IGNORECASE):
#                         teseracto = True
#                         print('Ha seleccionado dibujar un teseracto')

#         except sr.RequestError:
#             print("API no disponible")
#         except sr.UnknownValueError:
#             print("")
