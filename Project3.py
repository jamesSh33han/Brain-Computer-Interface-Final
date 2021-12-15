# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:36:55 2021

Project3.py

File that defines functions load_data, get_eeg_epochs, get_truth_event_labels, plot_power_spectrum, perform_ICA.
These functions load in a specified subjects raw EEG data file from the OPENMIIR dataset, epochs the EEG data into
target/nontarget epochs, defines truth event labels from the epoched data (target = true, nontarget = false), calculates
and plots the target/nontarget mean power spectrum for a specified channel, and computes ICA on the EEG data

@author: spenc, JJ
"""
#%% Import Statements
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Define figure size
plt.rcParams["figure.figsize"] = (14,8)

#%% Loading in and pre-processing the data
def load_data(subject):
    '''
    Function to load in the specified subjects .fif data file

    Parameters
    ----------
    subject : string of subject number (two digits)
        String denoting which subject we are analyzing.

    Returns
    -------
    fif_file : Raw MNE FIF file
        FIF file with MNE built in functions - all data can be extracted.
    raw_eeg_data : Array of size (channels, samples) - samples depend on which subject is read
        Array of floats containing the eeg recording data taken at each time point (across all
        trials and condiditons) for the one subject.
    eeg_times : Array of time points eeg samples were taken
        1-D array of times (in seconds) of the time points each eeg sample was taken at.
    channel_names: Array of channel names (each is string)
        Array of eeg channel names.
    fs : float
        smapling frquency of 512 Hz.

    '''

    fif_file=mne.io.read_raw_fif(f'data/P{subject}-raw.fif', preload=True)
    
    # pre-processing data before extraction
    print('Band-pass filtering between 1 - 30 Hz...')
    fif_file.filter(1,30)
    print('Rereferencing the raw data to the average across electrodes...')
    fif_file.set_eeg_reference(ref_channels='average')
    
    # extracting data
    raw_eeg_data = fif_file.get_data()[0:64, :]
    channel_names = fif_file.ch_names[0:64]
    eeg_times = fif_file.times
    fs = fif_file.info['sfreq']
    channel_names = np.array(channel_names)
    return fif_file, raw_eeg_data, eeg_times, channel_names, fs
    


#%% Epoching the data
def get_eeg_epochs(fif_file, raw_eeg_data, start_time, end_time, fs):
    '''
    Function to epoch the EEG raw data into target/nontarget epochs. 

    Parameters
    ----------
    fif_file : Raw MNE FIF file
        FIF file with MNE built in functions - all data can be extracted.
    raw_eeg_data : Array of size (channels, samples) - samples depend on which subject is read
        Array of floats containing the eeg recording data taken at each time point (across all
        trials and condiditons) for the one subject.
    start_time : float
        start time relative to event start.
    end_time : float
        end time relative to event start.
    fs : float
        smapling frquency of 512 Hz.

    Returns
    -------
    eeg_epochs : 3-D Array of size (trials, channels, time points)
        3-D array contianing epoched eeg data into the trials seen in the experiment.
    epoch_times : 1-D array of length epoch time points
        Array of times using epoch time points.
    target_events : array of size (target_events, (event onset, post-experiment feedback, stimulus/condiiton id))
        array containing the information on the target events (where the participant was listeing to music).
    all_trials : array of size (all trials, (event onset, post-experiment feedback, stimulus/condiiton id))
        array containing the information on all events.

    '''
    eeg_epochs = np.array([])
    all_trials = mne.find_events(fif_file)
    all_trials = all_trials[all_trials[:, 2] <1000]
    # target_events = all_trials[np.logical_not(np.logical_and(all_trials[:,2] > 20, all_trials[:,2] > 999))]
    
    
    # event_stimulus_ids = []
    # for event_index in range(len(target_events)):
    #     stimulus_id = target_events[event_index, 2]//10
    #     event_stimulus_ids.append(stimulus_id)
        
    event_start_times = all_trials[:, 0]
    for event_start_time in event_start_times:
        start_epoch = int(event_start_time) - int(start_time*fs)
        end_epoch = int(int(event_start_time) + (end_time*fs))
        epoch_data = raw_eeg_data[:, start_epoch:end_epoch]
        eeg_epochs = np.append(eeg_epochs, epoch_data)
    eeg_epochs = np.reshape(eeg_epochs, [len(all_trials), np.size(raw_eeg_data, axis=0), int(end_time*fs)])
    epoch_times = np.arange(0, np.size(eeg_epochs, axis=2))
    return eeg_epochs, epoch_times, all_trials

#%% Setting Event Truth Labels
def get_event_truth_labels(all_trials):
    '''
    Function to go through the data from every event and label each event as a target event (true) or a nontarget event (false)

    Parameters
    ----------
    all_trials : array of size (all trials, (event onset, post-experiment feedback, stimulus/condiiton id))
        array containing the information on all events (both perceived and imagined trials).

    Returns
    -------
    is_target_event: boolean array
        boolean array containing labels denoting weather trial contained a perceived music (target) event or not.

    '''
    is_target_event = np.array([])
    for trial_index in range(len(all_trials)):
        event_id = all_trials[trial_index, 2]
        condition = event_id % 10
        if condition == 1:
            is_target_event = np.append(is_target_event,True)
        elif condition == 2 or condition==3 or condition == 4:
            is_target_event = np.append(is_target_event,False)
        else:
            pass
    is_target_event = np.array(is_target_event, dtype='bool')
    return is_target_event



#%% Plotting Mean Power Spectrum
def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_target_event, channels_to_plot, channel_names):
    '''
    Function to plot calculated target/nontarget events mean power spectrum for specified channels
    
    Parameters
    ----------
    eeg_epochs_fft : Array of complex128
        3-D array holding epoched data in the frequency domain for each trial.
    fft_frequencies : Array of float64
        Array containing the frequency corresponding to each column of the Fourier transform data.
    is_target_event : 1-D boolean array 
        Boolean array representing trials the subject perceived music vs imagined music.
    channels_to_plot : list 
        list of channels we wish to plot the raw data for.
    channels : Array of str128
        List of channel names from original dataset.

    Returns
    -------
    None.

    '''
    target_trials = eeg_epochs_fft[is_target_event]
    non_target_trials = eeg_epochs_fft[~is_target_event]
    
    # Calculate mean power spectra
    mean_target_trials = np.mean(abs(target_trials), axis=0)**2
    mean_non_target_trials = np.mean(abs(non_target_trials), axis=0)**2
    
    mean_power_spectrum_target = mean_target_trials/mean_target_trials.max(axis=1, keepdims=True)
    mean_power_spectrum_nontarget = mean_non_target_trials/mean_non_target_trials.max(axis=1, keepdims=True)

    power_in_db_target = 10*np.log10(mean_power_spectrum_target)
    power_in_db_nontarget = 10*np.log10(mean_power_spectrum_nontarget)

    # Plot mean power spectrum of 12 and 15 Hz trials
    for channel_index, channel in enumerate(channels_to_plot):
        index_to_plot = np.where(channel_names==channel)[0][0]
        ax1=plt.subplot(len(channels_to_plot), 1, channel_index+1)
        plt.plot(fft_frequencies,power_in_db_target[index_to_plot], label='target', color='red')
        plt.plot(fft_frequencies,power_in_db_nontarget[index_to_plot], label='nontarget', color='green')
        plt.legend()
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.tight_layout()

        plt.grid()
    plt.savefig(f'figures/MeanPowerSpectrumChannel{channel}.png')

    
#%% ICA
def perform_ICA(raw_fif_file, channel_names, top_n_components):
    '''
    Function to preform ICA on the specified raw EEG data

    Parameters
    ----------
    fif_file : Raw MNE FIF file
        FIF file with MNE built in functions - all data can be extracted.
    channel_names: Array of channel names (each is string)
        Array of eeg channel names.
    top_n_components : int
        the number of top components the user wishes to plot.

    Returns
    -------
    ica : 
        object that contains ICA data

    '''
    picks_eeg = mne.pick_types(raw_fif_file.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')[0:64]
    ica = mne.preprocessing.ICA(n_components=64, random_state=97, max_iter=800)
    ica.fit(raw_fif_file, picks=picks_eeg, decim=3, reject=dict(mag=4e-12, grad=4000e-13))
    mixing_matrix = ica.mixing_matrix_
    ica.plot_components(picks = np.arange(0,top_n_components))
    plt.figure('Topo')
    plt.savefig(f'figures/Top{top_n_components}ICA.png')

    return ica


def plot_component_variance(ica, components, eeg_epochs, is_target_event):
    
    mixing_matrix = ica.mixing_matrix_
    unmixing_matrix = ica.unmixing_matrix_
    source_activations = np.matmul(unmixing_matrix, eeg_epochs)
    plt.figure('variance hists')
    for component in components:
        plt.subplot(2,5,component+1)
        component_activation = source_activations[:, component, :]
        component_activation_variances = np.var(component_activation, axis = 1)
        
        target_activation_vars = component_activation_variances[is_target_event]
        nontarget_activation_vars = component_activation_variances[~is_target_event]
        nontarget_activation_vars = np.delete(nontarget_activation_vars, 178)
        
        
        plt.hist([target_activation_vars, nontarget_activation_vars], label=['Perception', 'Imagination'])
    return source_activations
# %%

def make_prediction(source_activations, component, is_target_event, threshold):
    component_activation = source_activations[:, component, :]
    component_activation_variances = np.var(component_activation, axis = 1)
    
    # target_activation_vars = component_activation_variances[is_target_event]
    # nontarget_activation_vars = component_activation_variances[~is_target_event]
    # nontarget_activation_vars = np.delete(nontarget_activation_vars, 178)
    
    # plt.hist([target_activation_vars, nontarget_activation_vars], label=['Perception', 'Imagination'])
    # plt.axvline(x=threshold)
    
    predicted_labels = [] 
    for variance in component_activation_variances:
        if variance >= threshold:
            predicted_labels.append(1)
            
        else:
            predicted_labels.append(0)
            
    return predicted_labels

def evaluate_predictions(predictions, truth_labels):
    accuracy = np.mean(predictions==truth_labels)
    cm = confusion_matrix(truth_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    return accuracy, cm, disp
    
    

def test_all_components_thresholds(components, source_activations, is_target_event):
    all_accuracies = np.array([])
    all_thresholds = np.array([])
    all_true_positive_percentages = np.array([])
    components = components[::-1]
    for component in components:
        component_activation = source_activations[:, component, :]
        component_activation_variances = np.var(component_activation, axis = 1)
        # delete 238th variance, very large relatively, outlier
        component_activation_variances = np.delete(component_activation_variances, 238)
        min_threshold = np.min(component_activation_variances)
        max_threshold = np.max(component_activation_variances)
        # creates an array of thresholds based on the components range of values
        thresholds = np.arange(min_threshold, max_threshold, (max_threshold-min_threshold)/10)
        all_thresholds = np.append(all_thresholds, thresholds)
        for threshold in thresholds:
            predicted_labels = make_prediction(source_activations, component, is_target_event, threshold)
            accuracy, cm, disp = evaluate_predictions(predicted_labels, is_target_event*1)
            all_accuracies = np.append(all_accuracies, accuracy)
            tp_percent = cm[1][1]/60
            all_true_positive_percentages = np.append(all_true_positive_percentages, tp_percent)
    all_accuracies = np.reshape(all_accuracies, (len(thresholds), len(components)))        
    all_thresholds = np.reshape(all_thresholds, (len(thresholds), len(components)))
    all_true_positive_percentages = np.reshape(all_true_positive_percentages, (len(thresholds), len(components)))
    plt.subplot(1, 2, 1)
    plt.imshow(all_accuracies, extent = (components[-1], components[0], components[-1], components[0]))
    plt.colorbar(label = 'Accuracy (% Correct)', fraction=0.046, pad=0.04)
    # plt.subplot(1, 3, 2)
    # plt.plot(all_)
    plt.subplot(1, 2, 2)
    plt.imshow(all_true_positive_percentages, extent = (components[-1], components[0], components[-1], components[0]))
    plt.colorbar(label = 'TP %', fraction=0.046, pad=0.04)
    return all_accuracies, all_thresholds, all_true_positive_percentages
        

    
    
