# -*- coding: utf-8 -*-
"""
import_ssvep_data.py

Lab 3: Frequency Space
- Loads raw EEG data
- Plots specified raw electrode data
- Epochs EEG data
- Converts EEG data into frequency domain
- Plots power spectra for specified channels

Created on Thu Oct 14 12:21:20 2021

@author: spenc
"""
# %% Import Packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft


# %% Part 1: Load the Data
def load_ssvep_data(subject, data_directory):
    '''
    Function that loads the SSVEP data and returns the data as a dictionary with each field holding data

    Parameters
    ----------
    subject : int
        Specifies which subject we are looking at.
    data_directory : string
        string holding the path to where the data is stored.

    Returns
    -------
    data_dict : dictionary
        Dictionary with 6 fields each holding an array of data.

    '''
    # Load data and convert into dictionary
    data_dict = dict(np.load(data_directory+f'SSVEP_S{subject}.npz', allow_pickle=True))
    return data_dict

# %% Part 2: Plot the Data
def plot_raw_data(data_dict, subject, channels_to_plot):
    '''
    Function that plots the raw eeg data for a ceratin subject on channels the user specifies

    Parameters
    ----------
    data_dict : dictionary
        Dictionary with 6 fields each holding an array of data.
    subject : int
        Specifies which subject we are looking at.
    channels_to_plot : list 
        list of channels we wish to plot the raw data for.

    Returns
    -------
    None.

    '''
    # Select channel data
    channels = data_dict['channels']
    for channel in channels_to_plot:
        index_to_plot = np.where(channels==channel)[0][0]
        
        # Plot frequency data
        ax1=plt.subplot(2, 1, 1)
        for i in range(len(data_dict['event_types'])):
            start_time = data_dict['event_samples'][i]/data_dict['fs']
            end_time = (data_dict['event_samples'][i]+data_dict['event_durations'][i])/data_dict['fs']
            plt.scatter(start_time, data_dict['event_types'][i], color='blue')
            plt.scatter(end_time, data_dict['event_types'][i], color='blue')
            plt.plot([start_time, end_time],[data_dict['event_types'][i],data_dict['event_types'][i]], color='blue', markevery=10)
            plt.title(f'SSVEP Subject {subject} Raw Data')
            plt.xlabel('Time (s)')
            plt.ylabel('Flash Frequency')
            plt.grid()
        
        # Plot voltage data
        times = np.arange(0, np.size(data_dict['eeg'], axis=1)/int(data_dict['fs']), step=1/int(data_dict['fs']))
        ax2=plt.subplot(2, 1, 2, sharex=ax1)
        plt.plot(times,data_dict['eeg'][index_to_plot]*1000000, label = f'{channel}')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (µV)')
        plt.savefig(f'SSVEP_S{subject}.png')
        plt.grid()
        plt.tight_layout()
    plt.clf()


# %% Part 3: Extract the Epochs
def epoch_ssvep_data(data_dict, epoch_start_time=0, epoch_end_time=20):
    '''
    Function that epochs ssvep data based on an epoch start and end time. Epochs data into 'trials' based on when events occur to create 3-D epoched array

    Parameters
    ----------
    data_dict : dictionary
        Dictionary with 6 fields each holding an array of data.
    epoch_start_time : int
        Integer representing the start time in seconds we want to start epoching at. The default is 0.
    epoch_end_time : int
        Integer representing the end time in seconds we want to end epoching at. The default is 20.

    Returns
    -------
    eeg_epochs : 3-D Array of float64 of size (trials, channels, time points)
        Array holding the epoched data. The data is epoched to have each trial contain data for evey channel during the duration of an event.
    epoch_times : 1-D array of float64
        Array of float64 holding time points of eeg samples.
    is_trial_15Hz : 1-D boolean array 
        Boolean array representing trials in which flashing at 15 Hz occurred.

    '''
    eeg_data = data_dict['eeg'] * 1000000
    event_durations = data_dict['event_durations']
    event_samples = data_dict['event_samples']
    event_type = data_dict['event_types']
    fs = data_dict['fs']
    # event_start_times = event_samples/fs
    eeg_epochs = np.array([])
    for sample_index in event_samples:
        start_epoch = int(sample_index+(epoch_start_time*fs))
        end_epoch = int(start_epoch+(epoch_end_time-epoch_start_time)*fs)
        epoch_data = eeg_data[:, start_epoch:end_epoch]
        # append the epoch to the eeg_epochs array from before
        eeg_epochs = np.append(eeg_epochs, epoch_data)    
    eeg_epochs = np.reshape(eeg_epochs, [len(event_samples),len(data_dict['channels']), int((epoch_end_time-epoch_start_time)*fs)])
    epoch_times = np.arange(epoch_start_time, epoch_end_time, step = 1/fs)
    
    
    is_trial_15Hz = [event_type[:] == '15hz']
    
    return eeg_epochs, epoch_times, is_trial_15Hz[0]
    

# %% Part 4: Take the Fourier Transform
def get_frequency_spectrum(eeg_epochs, fs):
    '''
    Function that takes epoched data in the time domain and takes the Fourier transform of each
    epoch, to convert into the frequency spectrum for each trial.

    Parameters
    ----------
    eeg_epochs : 3-D Array of float64 of size (trials, channels, time points)
        Array holding the epoched data. The data is epoched to have each trial contain data for evey channel during the duration of an event.
    fs : Array of float64
        Sampling frequency.

    Returns
    -------
    eeg_epochs_fft : Array of complex128
        3-D array holding epoched data in the frequency domain for each trial.
    fft_frequencies : Array of float64
        Array containing the frequency corresponding to each column of the Fourier transform data.

    '''
    # Take fast fourier transform
    eeg_epochs_fft = np.fft.rfft(eeg_epochs)
    
    # Calculate corresponding frequencies
    fft_frequencies = np.fft.rfftfreq(np.size(eeg_epochs, axis=2), d=1/fs) 
    
    return eeg_epochs_fft, fft_frequencies

# %% Part 5: Plot the Power Spectra   
def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15Hz, channels_to_plot, channels):
    '''
    

    Parameters
    ----------
    eeg_epochs_fft : Array of complex128
        3-D array holding epoched data in the frequency domain for each trial.
    fft_frequencies : Array of float64
        Array containing the frequency corresponding to each column of the Fourier transform data.
    is_trial_15Hz : 1-D boolean array 
        Boolean array representing trials in which flashing at 15 Hz occurred.
    channels_to_plot : list 
        list of channels we wish to plot the raw data for.
    channels : Array of str128
        List of channel names from original dataset.

    Returns
    -------
    None.

    '''
    # Differentiate 12 Hz and 15 Hz trials
    eeg_trials_12Hz = eeg_epochs_fft[~is_trial_15Hz]
    eeg_trials_15Hz = eeg_epochs_fft[is_trial_15Hz]
    
    # Calculate mean power spectra
    mean_power_spectrum_12Hz = np.mean(abs(eeg_trials_12Hz), axis=0)**2
    mean_power_spectrum_15Hz = np.mean(abs(eeg_trials_15Hz), axis=0)**2
    
    # Normalize spectrum
    mean_power_spectrum_12Hz_norm = mean_power_spectrum_12Hz/mean_power_spectrum_12Hz.max(axis=1, keepdims=True)
    mean_power_spectrum_15Hz_norm = mean_power_spectrum_15Hz/mean_power_spectrum_15Hz.max(axis=1, keepdims=True)

    # Convert to decibels
    power_in_db_12Hz = 10*np.log10(mean_power_spectrum_12Hz_norm)
    power_in_db_15Hz = 10*np.log10(mean_power_spectrum_15Hz_norm)

    # Plot mean power spectrum of 12 and 15 Hz trials
    for channel_index, channel in enumerate(channels_to_plot):
        index_to_plot = np.where(channels==channel)[0][0]
        ax1=plt.subplot(len(channels_to_plot), 1, channel_index+1)
        plt.plot(fft_frequencies,power_in_db_12Hz[index_to_plot], label='12Hz', color='red')
        plt.plot(fft_frequencies,power_in_db_15Hz[index_to_plot], label='15Hz', color='green')
        plt.legend()
        plt.title(f'Mean {channel} Frequency Content for SSVEP Data')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.tight_layout()
        plt.vlines(12,-100,0,colors='red',linestyles='dotted')
        plt.vlines(15,-100,0,colors='green',linestyles='dotted')
        plt.grid()

    
    




    
    
    
    
    
    
    
    
    
    
    
    