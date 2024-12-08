# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import mode

from features import *

def bandpass_filter(data, fs=2000.0, lowcut=20.0, highcut=450.0, order=4):
    """Apply a bandpass filter and remove powerline noise and its harmonics."""
    # Apply bandpass filter
    sos = butter(order, [lowcut, highcut], fs=fs, btype='bandpass', output='sos')
    filtered_data = sosfiltfilt(sos, data.T).T

    # Apply notch filters to remove powerline noise and its harmonics
    powergrid_noise_frequencies_Hz = [harmonic_idx * 50 for harmonic_idx in range(1, 3)]  # 50Hz and 100Hz
    for noise_frequency in powergrid_noise_frequencies_Hz:
        sos = butter(order, [noise_frequency - 2, noise_frequency + 2], fs=fs, btype='bandstop', output='sos')
        filtered_data = sosfiltfilt(sos, filtered_data.T).T

    return filtered_data

def extract_features_advanced(trial, deadzone, winsize, wininc):
    """Define a function to extract all features from a single trial."""
    feat_mav = getmavfeat(trial, winsize, wininc)
    feat_mavs = getmavsfeat(trial, winsize, wininc)
    feat_rms = getrmsfeat(trial, winsize, wininc)
    feat_zc = getzcfeat(trial, deadzone, winsize, wininc)
    feat_ssc = getsscfeat(trial, deadzone, winsize, wininc)
    feat_wl = getwlfeat(trial, winsize, wininc)
    feat_mDWT = getmDWTfeat(trial, winsize, wininc)
    feat_iav = getiavfeat(trial, winsize, wininc)
    feat_TD = getTDfeat(trial, deadzone, winsize, wininc)
    edges = np.arange(-3, 3.3, 0.3)
    feat_HIST = getHISTfeat(trial, winsize, wininc, edges)

    # Concatenate all features
    feat = np.hstack((feat_mav, feat_mavs, feat_rms, feat_zc, feat_ssc, feat_wl, feat_mDWT, feat_iav, feat_TD, feat_HIST))

    return feat

# Frequency domain features
def mean_frequency(x, fs=2000):
    """Calculate the mean frequency of the signal."""
    freqs, psd = welch(x, fs=fs, axis=0)
    freqs = freqs[:, np.newaxis]
    mean_freq = np.sum(freqs * psd, axis=0) / np.sum(psd, axis=0)
    return mean_freq.mean() if mean_freq.size > 1 else mean_freq

def median_frequency(x, fs=2000):
    """Calculate the median frequency of the signal."""
    freqs, psd = welch(x, fs=fs, axis=0)
    cumulative_power = np.cumsum(psd, axis=0)
    if psd.ndim == 1:
        total_power = cumulative_power[-1]
        median_freq = freqs[np.where(cumulative_power >= total_power / 2)[0][0]]
    else:
        total_power = cumulative_power[-1, :]
        median_freqs = np.zeros(psd.shape[1])
        for i in range(psd.shape[1]):
            median_freqs[i] = freqs[np.where(cumulative_power[:, i] >= total_power[i] / 2)[0][0]]
        median_freq = median_freqs.mean() if median_freqs.size > 1 else median_freqs
    return median_freq

def total_power(x, fs=2000):
    """Calculate the total power of the signal."""
    freqs, psd = welch(x, fs=fs, axis=0)
    total_pwr = np.sum(psd, axis=0)
    return total_pwr.mean() if total_pwr.size > 1 else total_pwr

# Build dataset from NinaPro data
def build_dataset_from_ninapro(emg, stimulus, repetition, features=None):
    """Build a dataset from the NinaPro database."""
    # Calculate the number of unique stimuli and repetitions, subtracting 1 to exclude the resting condition
    n_stimuli = np.unique(stimulus).size - 1
    n_repetitions = np.unique(repetition).size - 1
    # Total number of samples is the product of stimuli and repetitions
    n_samples = n_stimuli * n_repetitions
    
    # Number of channels in the EMG data
    n_channels = emg.shape[1]
    # Calculate the total number of features by summing the number of channels for each feature
    n_features = n_channels * len(features)
    
    # Initialize the dataset and labels arrays with zeros
    dataset = np.zeros((n_samples, n_features))
    labels = np.zeros(n_samples)
    current_sample_index = 0
    
    # Loop over each stimulus and repetition to extract features
    for i in range(n_stimuli):
        for j in range(n_repetitions):
            # Assign the label for the current sample
            labels[current_sample_index] = i + 1
            # Calculate the current sample index based on stimulus and repetition
            current_sample_index = i * n_repetitions + j
            current_feature_index = 0
            # Select the time steps corresponding to the current stimulus and repetition
            selected_tsteps = np.logical_and(stimulus == i + 1, repetition == j + 1).squeeze()
            
            # Loop over each channel
            for ch in range(n_channels):
                # Loop over each feature function provided
                for feature in features:
                    # Apply the feature function to the selected EMG data for the current channel and store the result
                    dataset[current_sample_index, current_feature_index] = feature(emg[selected_tsteps, ch])
                    # Update the feature index for the next feature
                    current_feature_index += 1

            # Move to the next sample
            current_sample_index += 1
            
    # Return the constructed dataset and corresponding labels
    return dataset, labels