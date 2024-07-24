import os
import numpy as np
import librosa
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('alphabet_recognition_model.h5')

# Define the labels (alphabet letters)
labels = sorted(os.listdir('data/alphabet_sounds/'))

# Function to preprocess the audio input
def preprocess_audio(file_path):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=None)
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    mfcc = np.expand_dims(mfcc, axis=-1) # Add channel dimension
    return mfcc

# Function to predict the letter
def predict_letter(file_path):
    mfcc = preprocess_audio(file_path)
    prediction = model.predict(mfcc)
    predicted_label_index = np.argmax(prediction)
    predicted_label = labels[predicted_label_index]
    return predicted_label

# Example usage
if __name__ == "__main__":
    file_path = input("Enter the path to the audio file: ")
    if os.path.exists(file_path):
        letter = predict_letter(file_path)
        print(f"The predicted letter is: {letter}")
    else:
        print("File not found.")
