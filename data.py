import os
from gtts import gTTS
from pydub import AudioSegment
import random

def modify_audio(file_path, output_path, speed=1.0, pitch=0):
    sound = AudioSegment.from_file(file_path)
    if speed != 1.0:
        sound = sound.speedup(playback_speed=speed)
    if pitch != 0:
        sound = sound._spawn(sound.raw_data, overrides={'frame_rate': int(sound.frame_rate * (2.0 ** (pitch / 12.0)))})
        sound = sound.set_frame_rate(sound.frame_rate)
    sound.export(output_path, format="wav")

def generate_alphabet_sounds(data_dir, num_samples=5):
    os.makedirs(data_dir, exist_ok=True)
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    for letter in alphabet:
        letter_dir = os.path.join(data_dir, letter)
        os.makedirs(letter_dir, exist_ok=True)
        for i in range(num_samples):
            tts = gTTS(text=letter, lang='en')
            temp_path = os.path.join(letter_dir, f'temp_sound{i + 1}.wav')
            tts.save(temp_path)

            speed = random.choice([0.8, 1.0, 1.2])
            pitch = random.choice([-2, 0, 2])
            output_path = os.path.join(letter_dir, f'sound{i + 1}.wav')
            modify_audio(temp_path, output_path, speed=speed, pitch=pitch)

            os.remove(temp_path)

def main():
    data_dir = 'data/alphabet_sounds'
    generate_alphabet_sounds(data_dir)

if __name__ == "__main__":
    main()
