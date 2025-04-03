import sounddevice as sd
import numpy as np
import requests
import io
import pyttsx3
from pydub import AudioSegment
import time

SERVER_URL = "http://localhost:8000"

# --- TTS Setup ---
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Slower speech
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)  # Change voice index if needed


def record_audio(duration=5, fs=16000):
    print("\n[Listening...]")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    return audio.flatten()


def speak(text: str):
    print(f"[AI]: {text}")
    engine.say(text)
    engine.runAndWait()


def main():
    # Register user
    name = input("Enter your name: ")
    requests.post(f"{SERVER_URL}/register_user", json={"name": name})
    speak(f"Hello {name}! I'm your mental health companion. How can I help you today?")

    while True:
        try:
            # Record audio
            audio = record_audio()
            audio_bytes = audio.tobytes()

            # Send to server
            files = {"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
            start_time = time.time()
            response = requests.post(f"{SERVER_URL}/process_audio", files=files)

            if response.status_code == 200:
                ai_text = response.json()["text"]
                speak(ai_text)
            else:
                speak("Sorry, I encountered an error.")

        except KeyboardInterrupt:
            speak("Goodbye!")
            break


if __name__ == "__main__":
    main()