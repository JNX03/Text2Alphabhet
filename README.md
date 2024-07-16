# Text 2 Alphabhet [CNN]

This project uses Convolutional Neural Networks (CNN) to recognize alphabet sounds. It consists of two main scripts: `train.py` for training the model and `main.py` for using the trained model to make predictions.

## Prerequisites

libraries installed:
```bash
pip install tensorflow numpy librosa matplotlib
```

## Dataset Structure (for training)

Organize your dataset as follows:
```
data/alphabet_sounds/
    a/
        sound1.wav
        sound2.wav
        ...
    b/
        sound1.wav
        sound2.wav
        ...
    ...
    z/
        sound1.wav
        sound2.wav
        ...
```

## Training the Model

To train the model, run:
```bash
python train.py
```


This will train the model using the audio files and save it as `alphabet_model.h5`.

## Using the Trained Model

To use the trained model for making predictions, run:
```bash
python main.py --audio_path file.wav
```
Replace `file.wav` with the path to your test audio file.

## Notes

- Ensure your dataset is well-organized, with subdirectories for each alphabet (e.g., `data/alphabet_sounds/a`, `data/alphabet_sounds/b`, etc.).
- Adjust the paths (`data_dir` and `audio_path`) in the scripts according to your dataset's location and the test audio file's location.

<center>Made with ❤️</center>
