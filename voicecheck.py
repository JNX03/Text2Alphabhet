import pyttsx3

def list_available_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for index, voice in enumerate(voices):
        print(f"Voice {index}: {voice.name} ({voice.id})")

if __name__ == "__main__":
    list_available_voices()
