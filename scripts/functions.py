# Import necessary libraries
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import mode

from features import *

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Define a function to apply a bandpass filter to the data."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

def reshape_data(data, samples_per_trial):
    """Reshape the data into trials."""
    num_trials = data.shape[0] // samples_per_trial
    data_reshaped = data[:num_trials * samples_per_trial].reshape(num_trials, samples_per_trial, data.shape[1])
    return data_reshaped

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

def extract_features_basic(trial):
    num_channels = trial.shape[1]
    features = []

    for ch in range(num_channels):
        channel_data = trial[:, ch]

        # Mean Absolute Value (MAV)
        mav = np.mean(np.abs(channel_data))
        # Mean Absolute Value Slope (MAVS)
        mavs = np.mean(np.abs(np.diff(channel_data)))
        # Root Mean Square (RMS)
        rms = np.sqrt(np.mean(channel_data**2))
        # Zero Crossing (ZC)
        zc = np.sum(np.diff(np.sign(channel_data)) != 0)
        # Waveform Length (WL)
        wl = np.sum(np.abs(np.diff(channel_data)))
        # Slope Sign Changes (SSC)
        ssc = np.sum(np.diff(np.sign(np.diff(channel_data))) != 0)

        features.extend([mav, mavs, rms, zc, wl, ssc])

    return np.array(features)

def extract_features_from_data(data):
    """Define a function to extract all features from all trials."""
    num_trials = data.shape[0]
    features = []

    for i in range(num_trials):
        trial_features = extract_features_basic(data[i, :, :])
        features.append(trial_features)

    return np.vstack(features)

def reshape_labels(labels, samples_per_trial):
    """Reshape the labels to match the number of trials."""
    num_trials = labels.shape[0] // samples_per_trial
    labels_reshaped = labels[:num_trials * samples_per_trial].reshape(num_trials, samples_per_trial)
    labels_mode = mode(labels_reshaped, axis=1)[0].flatten()
    return labels_mode