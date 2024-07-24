import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define the data directory
DATA_DIR = "data/alphabet_sounds/"

# Function to load audio files and extract features
def load_data(data_dir):
    X = []
    y = []
    labels = sorted(os.listdir(data_dir))
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    
    for label in labels:
        files = os.listdir(os.path.join(data_dir, label))
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(data_dir, label, file)
                # Load audio file and extract features (MFCC)
                audio, sr = librosa.load(file_path, sr=None)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # Increased n_mfcc to 40
                mfcc = np.mean(mfcc.T, axis=0)
                X.append(mfcc)
                y.append(label_to_index[label])
    
    X = np.array(X)
    y = np.array(y)
    return X, y, labels

# Load and preprocess the data
X, y, labels = load_data(DATA_DIR)
X = np.expand_dims(X, -1)  # Add an extra dimension for the CNN

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, num_classes=len(labels))
y_test = to_categorical(y_test, num_classes=len(labels))

# Build the CNN model with 1D convolutional layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(40, 1)),  # Updated input shape
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=4096, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")

# Save the model
model.save('alphabet_recognition_model.h5')
