import os
import shutil
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from Train import *

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1, 8):
        signal = notch(signal, freq * harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal)) / old_freq
    sample_times = np.arange(0, times[-1], 1 / new_freq)
    result = np.interp(sample_times, times, signal)
    return result

def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:, i], *args, **kwargs))
    return np.stack(results, 1)

def preprocess_all_emg(base_dir, output_dir):
    preprocessed_emgs = []
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(base_dir):
        if filename.endswith(".npy"):
            raw_emg = np.load(os.path.join(base_dir, filename))

            # Apply notch filtering and remove drift
            x = apply_to_all(notch_harmonics, raw_emg, 60, 1000)
            x = apply_to_all(remove_drift, x, 1000)

            # Subsample to desired frequency
            emg = apply_to_all(subsample, x, 516.79, 1000)

            preprocessed_emgs.append(emg)
            # Save preprocessed EMG data
            save_filename = os.path.join(output_dir, f"{filename}")
            np.save(save_filename, emg)
            #print(f"Preprocessed EMG saved to:", save_filename)

    return preprocessed_emgs

# Example usage:
base_dir = "./from"
output_dir = "./to"
preprocessed_emgs = preprocess_all_emg(base_dir, output_dir)

# Define the path to the silent speech data file
silent_speech_file = './to/test.npy'  # Replace with your actual file path

# Load the trained model
model_path = './model.pth'
model = EMGModel(input_size, hidden_size, output_size)  # Assuming you have defined these variables
model.load_state_dict(torch.load(model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set the model to evaluation mode

# Function to preprocess and convert a single silent speech sample to a tensor
def preprocess_silent_speech(silent_speech_sample, padding_value=0):
    silent_speech_tensor = torch.tensor(silent_speech_sample, dtype=torch.float32)
    silent_speech_tensor = silent_speech_tensor.to(device)

    return silent_speech_tensor.unsqueeze(0)  # Add a batch dimension

# Function to evaluate a single silent speech sample
def evaluate_silent_speech(silent_speech_file, label_encoder):
    silent_speech_sample = np.load(silent_speech_file)
    silent_speech_tensor = preprocess_silent_speech(silent_speech_sample)

    with torch.no_grad():
        outputs = model(silent_speech_tensor)
        _, predicted = torch.max(outputs.data, 1)  # Get the maximum value along dim 1

        # Access the predicted class index
        predicted_index = predicted.argmax().item()

        # Use the loaded label encoder to map the index back to the original class
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    return predicted_label

# Load the label encoder from the pickle file
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Predict the word for the silent speech sample
predicted_word = evaluate_silent_speech(silent_speech_file, label_encoder)

predicted_wo = os.path.splitext(predicted_word)[0]
print(predicted_wo)

with open('predicted_words.txt', 'a') as file:
    file.write(predicted_wo + '\n')

shutil.rmtree(output_dir)
