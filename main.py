import numpy as np
import tensorflow as tf
import librosa
import argparse

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_audio(model, audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    if spectrogram.shape[1] < 128:
        pad_width = 128 - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    spectrogram = spectrogram[:, :128]
    spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
    spectrogram = spectrogram / 255.0
    prediction = model.predict(spectrogram)
    return np.argmax(prediction)

def main():
    parser = argparse.ArgumentParser(description='Predict the alphabet sound.')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the audio file to predict.')
    args = parser.parse_args()

    model_path = 'alphabet_model.h5'
    audio_path = args.audio_path

    model = load_model(model_path)
    prediction = predict_audio(model, audio_path)
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    predicted_label = alphabet[prediction]

    print(f'The predicted alphabet is: {predicted_label}')

if __name__ == "__main__":
    main()
