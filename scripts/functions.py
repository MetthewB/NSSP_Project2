# Import necessary libraries
import os
import zipfile
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from scipy.signal import butter, sosfiltfilt, welch
from statsmodels.tsa.ar_model import AutoReg


def extract_and_load_mat_files(base_zip_path, extraction_dir, subject_num):
    """
    Extracts zip files and loads the specified .mat file for a given subject.
    
    Parameters:
        base_zip_path (str): The base path to the zip files.
        extraction_dir (str): The directory where the files will be extracted.
        subject_num (int): The subject number to process.
    
    Returns:
        dict: A dictionary containing the data from the .mat file.
    """
    # Loop through all subjects
    for num in range(1, 28):
        zip_file_path = os.path.join(base_zip_path, f's{num}.zip')
        
        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_dir)
        
        # Define the paths to the .mat files
        mat_file_path_e1 = os.path.join(extraction_dir, f'S{num}_A1_E1.mat')
        mat_file_path_e2 = os.path.join(extraction_dir, f'S{num}_A1_E2.mat')
        mat_file_path_e3 = os.path.join(extraction_dir, f'S{num}_A1_E3.mat')
        
        # Delete the S[subject number]_A1_E2.mat and S[subject number]_A1_E3.mat files
        if os.path.exists(mat_file_path_e2):
            os.remove(mat_file_path_e2)
        if os.path.exists(mat_file_path_e3):
            os.remove(mat_file_path_e3)
    
    # Now, specifically work on the specified subject's S_A1_E1.mat
    mat_file_path_e1 = os.path.join(extraction_dir, f'S{subject_num}_A1_E1.mat')
    
    # Load the .mat file using scipy.io.loadmat
    mat_data = sio.loadmat(mat_file_path_e1)
    
    # Print the keys of the loaded .mat file to see available variables
    print(f"Dataset variables: {mat_data.keys()}")
    
    # Extract the data and labels
    emg = mat_data['emg']
    stimulus = mat_data['restimulus']
    repetition = mat_data['rerepetition']
    
    return emg, stimulus, repetition


def bandpass_filter(data, fs=2000.0, lowcut=20.0, highcut=450.0, order=4, apply_notch=True):
    """
    Apply a bandpass filter to the input data and optionally remove powerline noise and its harmonics.

    Parameters:
        data (ndarray): The input signal data to be filtered.
        fs (float): The sampling frequency of the data. Default is 2000.0 Hz.
        lowcut (float): The lower cutoff frequency for the bandpass filter. Default is 20.0 Hz.
        highcut (float): The upper cutoff frequency for the bandpass filter. Default is 450.0 Hz.
        order (int): The order of the Butterworth filter. Default is 4.
        apply_notch (bool): Whether to apply notch filters to remove powerline noise at 50 Hz and its harmonics. Default is True.

    Returns:
        ndarray: The filtered data.
    """
    # Apply bandpass filter
    sos = butter(order, [lowcut, highcut], fs=fs, btype='bandpass', output='sos')
    filtered_data = sosfiltfilt(sos, data.T).T

    if apply_notch:
        # Apply notch filters to remove powerline noise and its harmonics
        powergrid_noise_frequencies_Hz = [harmonic_idx * 50 for harmonic_idx in range(1, 3)]  # 50Hz and 100Hz
        for noise_frequency in powergrid_noise_frequencies_Hz:
            sos = butter(order, [noise_frequency - 2, noise_frequency + 2], fs=fs, btype='bandstop', output='sos')
            filtered_data = sosfiltfilt(sos, filtered_data.T).T

    return filtered_data


def mav(x):
    """Mean Absolute Value."""
    return np.mean(np.abs(x), axis=0)

def mean(x):
    """Mean."""
    return np.mean(x, axis=0)

def std(x):
    """Standard Deviation."""
    return np.std(x, axis=0)

def maxav(x):
    """Maximum Absolute Value."""
    return np.max(np.abs(x), axis=0)

def rms(x):
    """Root Mean Square."""
    return np.sqrt(np.mean(x**2, axis=0))

def wl(x):
    """Waveform Length."""
    return np.mean(np.abs(np.diff(x, axis=0)), axis=0)

def ssc(x):
    """Slope Sign Change."""
    return np.mean(np.diff(np.sign(np.diff(x, axis=0)), axis=0) != 0, axis=0)

def dft_energy(x):
    """Energy in DFT."""
    return np.mean(np.abs(np.fft.fft(x, axis=0))**2, axis=0)

def median_frequency(x):
    """Median Frequency."""
    freqs = np.fft.fftfreq(x.shape[0])
    magnitudes = np.abs(np.fft.fft(x, axis=0))
    cumulative_sum = np.cumsum(magnitudes, axis=0)
    total_sum = cumulative_sum[-1]
    median_freq = freqs[np.searchsorted(cumulative_sum, total_sum / 2)]
    return median_freq

def total_power(x, fs=2000):
    """Total Power."""
    freqs, psd = welch(x, fs=fs, axis=0)
    total_pwr = np.sum(psd, axis=0)
    return total_pwr

def wilson_amplitude(x, threshold=1e-5):
    """Wilson Amplitude."""
    wamp = np.sum(np.abs(x) > threshold, axis=0)
    return wamp

def ar_coefficients(x):
    """4th-order Auto-Regressive Coefficients."""
    model = AutoReg(x, lags=4, old_names=False)
    ar_fit = model.fit()
    return ar_fit.params.tolist() 

def log_variance(x):
    """Log Variance."""
    #normaize the data bewteen 1 and 2
    x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0)) + 1
    return np.log(np.var(x, axis=0))


def build_dataset_from_ninapro(emg, stimulus, repetition, features=None):
    """
    Build a dataset from NinaPro EMG data by extracting specified features.

    Parameters:
        emg (ndarray): The EMG data.
        stimulus (ndarray): The stimulus labels.
        repetition (ndarray): The repetition indices.
        features (list): List of feature functions to apply to the EMG data.

    Returns:
        tuple: The constructed dataset and corresponding labels.
    """
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


def build_dataset_from_ninapro_all_subjects(subject_id_column, emg, stimulus, repetition, features=None):
    """
    Builds a dataset from the Ninapro dataset for all subjects.

    Parameters:
    subject_id_column (np.ndarray): Array containing the subject IDs.
    emg (np.ndarray): Array containing the EMG data.
    stimulus (np.ndarray): Array containing the stimulus data.
    repetition (np.ndarray): Array containing the repetition data.
    features (list, optional): List of features to be used. Defaults to None.

    Returns:
    tuple: A tuple containing:
        - dataset_all (np.ndarray): Concatenated dataset for all subjects.
        - labels_all (np.ndarray): Concatenated labels for all subjects.
        - id_column (np.ndarray): Array of subject IDs corresponding to each row in the dataset.
    """
    # Number of unique subjects
    nb_subjects = np.unique(subject_id_column).size
    
    # Initialize lists to store the concatenated data
    dataset_all = []
    labels_all = []
    id_column = []

    # Loop over each subject
    for subject_num in np.unique(subject_id_column):
        # Filter data for the current subject
        subject_mask = subject_id_column == subject_num
        emg_subject = emg[subject_mask, :]
        stimulus_subject = stimulus[subject_mask]
        repetition_subject = repetition[subject_mask]
        
        # Build dataset and labels for the current subject
        dataset, labels = build_dataset_from_ninapro(emg_subject, stimulus_subject, repetition_subject, features)
        
        # Append the data for the current subject to the lists
        dataset_all.append(dataset)
        labels_all.append(labels)
        id_column.extend([subject_num] * dataset.shape[0])

    # Concatenate the data for all subjects
    dataset_all = np.concatenate(dataset_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    id_column = np.array(id_column)

    return dataset_all, labels_all, id_column


def load_and_concatenate_subjects(base_path, extraction_dir, n_subjects=27, isZipped=True):
    """
    Extracts EMG data from all subjects, concatenates them into a single matrix,
    and returns the concatenated data along with labels and repetitions.

    Parameters:
        base_path (str): Path to the directory containing the zip or .mat files.
        extraction_dir (str): Path to the directory for extracting files (if isZipped is True).
        num_subjects (int): Number of subjects to process.
        isZipped (bool): Whether the input files are zipped or already extracted.

    Returns:
        tuple: Subject ID column, Concatenated EMG data, stimulus labels, and repetition indices.
    """
    emg_data_all = []
    stimulus_all = []
    repetition_all = []

    for subject_num in range(1, n_subjects + 1):
        if isZipped:
            # Paths to zip and extraction
            zip_file_path = os.path.join(base_path, f's{subject_num}.zip')
            
            # Extract the zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extraction_dir)
            
            # Paths to .mat files after extraction
            mat_file_path_e1 = os.path.join(extraction_dir, f'S{subject_num}_A1_E1.mat')
            mat_file_path_e2 = os.path.join(extraction_dir, f'S{subject_num}_A1_E2.mat')
            mat_file_path_e3 = os.path.join(extraction_dir, f'S{subject_num}_A1_E3.mat')
        else:
            # Paths to .mat files directly in the base path
            mat_file_path_e1 = os.path.join(base_path, f'S{subject_num}_A1_E1.mat')
            mat_file_path_e2 = os.path.join(base_path, f'S{subject_num}_A1_E2.mat')
            mat_file_path_e3 = os.path.join(base_path, f'S{subject_num}_A1_E3.mat')
        
        # Remove unnecessary .mat files
        if os.path.exists(mat_file_path_e2):
            os.remove(mat_file_path_e2)
        if os.path.exists(mat_file_path_e3):
            os.remove(mat_file_path_e3)
        
        # Load the relevant .mat file
        if os.path.exists(mat_file_path_e1):
            mat_data = sio.loadmat(mat_file_path_e1)
            
            # Append data to the lists
            emg_data_all.append(mat_data['emg'])
            stimulus_all.append(mat_data['stimulus'])
            repetition_all.append(mat_data['repetition'])
    
    # Concatenate all data

    # Initialize the subject_id_column with the correct shape (no initial zeros)
    subject_id_column = np.array([])  # Start with an empty array

    # Loop through each subject and append the subject ID
    for i in range(1, n_subjects + 1):
        subject_id_column = np.concatenate((subject_id_column, i * np.ones((emg_data_all[i - 1].shape[0], ))), axis=0)
    emg_data_all = np.concatenate(emg_data_all, axis=0)
    stimulus_all = np.concatenate(stimulus_all, axis=0)
    repetition_all = np.concatenate(repetition_all, axis=0)

    return subject_id_column, emg_data_all, stimulus_all, repetition_all


def plot_emg_envelopes(emg, id_column, stimulus, repetition, n_stimuli, n_repetitions, mov_mean_length=25, filtered=False):
    """
    Plots the EMG envelopes for each subject.

    Parameters:
    emg (ndarray): The EMG data.
    id_column (ndarray): Array of subject IDs.
    stimulus (ndarray): Array of stimulus identifiers.
    repetition (ndarray): Array of repetition identifiers.
    n_stimuli (int): Number of different stimuli.
    n_repetitions (int): Number of repetitions for each stimulus.
    mov_mean_length (int): Length of the moving average window. Default is 25.
    """
    mov_mean_weights = np.ones(mov_mean_length) / mov_mean_length

    # Number of subjects (adjust if needed)
    unique_subjects = np.unique(id_column)

    # Initialize a color map for distinct colors
    def colors(index):
        """Return a color based on the index."""
        color_list = plt.cm.tab10.colors  # A predefined colormap
        return color_list[index % len(color_list)]

    # Create a 9x3 layout for the plots
    fig, axes = plt.subplots(9, 3, figsize=(15, 30), constrained_layout=True)
    axes = axes.ravel()

    # Iterate over all subjects and plot their emg_envelopes
    for i, subject_num in enumerate(unique_subjects):
        # Filter data for the current subject
        subject_mask = id_column == subject_num
        emg_subject = emg[subject_mask, :]
        stimulus_subject = stimulus[subject_mask]
        repetition_subject = repetition[subject_mask]

        # Initialize the data structure for current subject
        emg_windows_subject = [[None for _ in range(n_repetitions)] for _ in range(n_stimuli)]
        emg_envelopes_subject = [[None for _ in range(n_repetitions)] for _ in range(n_stimuli)]

        # Compute emg_windows and emg_envelopes for the current subject
        for stimuli_idx in range(n_stimuli):
            for repetition_idx in range(n_repetitions):
                idx = np.logical_and(stimulus_subject == stimuli_idx + 1, repetition_subject == repetition_idx + 1).flatten()
                emg_windows_subject[stimuli_idx][repetition_idx] = emg_subject[idx, :]
                emg_envelopes_subject[stimuli_idx][repetition_idx] = convolve1d(emg_windows_subject[stimuli_idx][repetition_idx], mov_mean_weights, axis=0)

        # Extract the emg envelope for the first stimulus and first repetition
        emg_envelope_first_stimuli_repetition = emg_envelopes_subject[0][0]

        # Plot the envelope for the first stimulus and first repetition
        ax = axes[i]
        for channel_idx in range(emg_envelope_first_stimuli_repetition.shape[1]):  # Iterate over channels
            ax.plot(emg_envelope_first_stimuli_repetition[:, channel_idx], color=colors(channel_idx), label=f'Channel {channel_idx+1}')
        ax.set_title(f"Subject {int(subject_num)}")
        if i >= 24:  # Bottom row
            ax.set_xlabel("Time [s]", fontsize=10)
        if i % 3 == 0:  # Leftmost column
            ax.set_ylabel("EMG Signal [mV]", fontsize=10)
        plt.suptitle("Envelopes of the EMG signal for all subjects (Stimulus 1, Repetition 1)", fontsize=16)
        # Check for channels that are zero
        zero_channels = np.where(np.all(emg_envelope_first_stimuli_repetition == 0, axis=0))[0]

        # print the zero channels
        if zero_channels.size > 0: 
            print(f"Subject {int(subject_num)} has zero channels: {zero_channels}")

    # Add a legend to the first subplot
    axes[0].legend(loc='upper right', prop={'size': 8})
    figures_dir = os.path.join('..', 'output', 'Part 2')
    if filtered:
        plt.savefig(os.path.join(figures_dir, 'emg_envelopes_all_subjects_filtered.png'))
    else:
        plt.savefig(os.path.join(figures_dir, 'emg_envelopes_all_subjects.png'))
    plt.show()