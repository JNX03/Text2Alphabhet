import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Define the input and output directories
input_file = 'inbox-wav/ee8f9fa9-123f-3cff-f497-73a92be86d82.wav'
output_base_dir = 'folder'

# Ensure output directories exist
for letter in 'abcdefghijklmnopqrstuvwxyz':
    os.makedirs(os.path.join(output_base_dir, letter), exist_ok=True)

# Function to split audio on silence and save segments
def split_audio_on_silence(audio_file_path, output_dir):
    audio = AudioSegment.from_wav(audio_file_path)
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)

    letters = 'abcdefghijklmnopqrstuvwxyz'
    for i, chunk in enumerate(chunks):
        if i < len(letters):
            letter = letters[i]
            chunk.export(os.path.join(output_dir, letter, f"sound{i+11}.wav"), format="wav")

# Process the audio file
split_audio_on_silence(input_file, output_base_dir)

print("Audio files have been split and saved to corresponding directories.")
