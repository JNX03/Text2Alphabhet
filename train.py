import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import librosa
import matplotlib.pyplot as plt

def load_data(data_dir):
    data = []
    labels = []
    alphabet_dirs = os.listdir(data_dir)
    for label, alphabet in enumerate(alphabet_dirs):
        alphabet_path = os.path.join(data_dir, alphabet)
        for file in os.listdir(alphabet_path):
            file_path = os.path.join(alphabet_path, file)
            audio, sr = librosa.load(file_path, sr=None)
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            if spectrogram.shape[1] < 128:
                pad_width = 128 - spectrogram.shape[1]
                spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
            spectrogram = spectrogram[:, :128]
            data.append(spectrogram)
            labels.append(label)
    data = np.array(data)
    data = data[..., np.newaxis]
    return data, np.array(labels)

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.3),  # Adjusted dropout rate
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),  # Adjusted dropout rate
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),  # Adjusted dropout rate
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def main():
    data_dir = 'data/alphabet_sounds'
    data, labels = load_data(data_dir)
    data = data / 255.0

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    input_shape = x_train.shape[1:]
    num_classes = len(set(labels))

    model = create_model(input_shape, num_classes)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    ]

    history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=callbacks)
    plot_history(history)

if __name__ == "__main__":
    main()
