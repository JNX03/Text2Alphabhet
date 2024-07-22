import os
from gtts import gTTS
from pydub import AudioSegment
import pyttsx3

def convert_mp3_to_wav(mp3_path, wav_path):
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")

def add_silence(audio, duration=500):
    silence_segment = AudioSegment.silent(duration=duration)
    return audio + silence_segment

def save_pyttsx3_ssml(letter, voice, file_path, rate=130):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.setProperty('voice', voice)
    ssml = f'<speak><prosody rate="slow">{letter}</prosody></speak>'
    temp_wav_path = file_path.replace('.wav', '_temp.wav')
    engine.save_to_file(ssml, temp_wav_path)
    engine.runAndWait()
    
    audio = AudioSegment.from_wav(temp_wav_path)
    audio = add_silence(audio, duration=500)
    audio.export(file_path, format="wav")
    os.remove(temp_wav_path)  # Remove temporary wav file

def create_dataset():
    # Initialize the pyttsx3 engine to get available voices
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    # Directory structure
    base_dir = 'data/alphabet_sounds'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Generate 50 audio files for each letter with different voices
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        letter_dir = os.path.join(base_dir, letter)
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)

        for i in range(1, 11):
            file_path_gtts = os.path.join(letter_dir, f'gtts_sound{i}.wav')
            file_path_pyttsx3_male = os.path.join(letter_dir, f'pyttsx3_male_sound{i}.wav')
            file_path_pyttsx3_female = os.path.join(letter_dir, f'pyttsx3_female_sound{i}.wav')

            # Generate with gTTS
            if not os.path.exists(file_path_gtts):
                print(f'Generating {file_path_gtts}')
                tts = gTTS(text=letter.upper(), lang='en', slow=True)
                temp_mp3_path = os.path.join(letter_dir, f'temp{i}.mp3')
                tts.save(temp_mp3_path)
                audio = AudioSegment.from_mp3(temp_mp3_path)
                audio = add_silence(audio, duration=500)
                audio.export(file_path_gtts, format="wav")
                os.remove(temp_mp3_path)  # Remove temporary mp3 file

            # Generate with pyttsx3 (male voice)
            if not os.path.exists(file_path_pyttsx3_male):
                print(f'Generating {file_path_pyttsx3_male}')
                save_pyttsx3_ssml(letter.upper(), voices[0].id, file_path_pyttsx3_male)

            # Generate with pyttsx3 (female voice)
            if not os.path.exists(file_path_pyttsx3_female):
                print(f'Generating {file_path_pyttsx3_female}')
                save_pyttsx3_ssml(letter.upper(), voices[1].id, file_path_pyttsx3_female)
            else:
                print(f'Skipping {file_path_gtts}, {file_path_pyttsx3_male}, and {file_path_pyttsx3_female}, already exist')

if __name__ == '__main__':
    create_dataset()
