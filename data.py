import os
import requests

# Define the base URL for downloading the dataset
BASE_URL = "https://robertmassaioli.bitbucket.io/alphabet/"  # Replace with actual base URL
LETTERS = 'abcdefghijklmnopqrstuvwxyz'
DEST_PATH = 'data/alphabet_sounds'

# Function to download a file
def download_file(url, dest):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(dest, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {dest}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# Create directory structure
for letter in LETTERS:
    letter_path = os.path.join(DEST_PATH, letter)
    os.makedirs(letter_path, exist_ok=True)

# Example list of filenames for each letter (replace with actual filenames)
for letter in LETTERS:
    for i in range(1, 6):  # Adjust range for the number of files
        file_url = f"{BASE_URL}{letter}/sound{i}.wav"  # Construct URL
        dest_file = os.path.join(DEST_PATH, letter, f'sound{i}.wav')
        download_file(file_url, dest_file)
