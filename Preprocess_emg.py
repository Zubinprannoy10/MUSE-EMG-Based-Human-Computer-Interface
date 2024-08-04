import os
import numpy as np
import scipy.signal

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
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(base_dir):
        if filename.endswith(".npy"):
            raw_emg = np.load(os.path.join(base_dir, filename))

            x = apply_to_all(notch_harmonics, raw_emg, 60, 1000)
            x = apply_to_all(remove_drift, x, 1000)
            emg = apply_to_all(subsample, x, 516.79, 1000)

            preprocessed_emgs.append(emg)

            save_filename = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npy")
            np.save(save_filename, emg)
            print(f"Preprocessed EMG saved to: {save_filename}")

    return preprocessed_emgs

# Example usage:
base_dir = "./Data"
output_dir = "./Preprocess"
preprocessed_emgs = preprocess_all_emg(base_dir, output_dir)

