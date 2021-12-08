# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:36:55 2021

@author: spenc
"""
#%%
import mne
from mne.io.pick import channel_type
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

plt.rcParams["figure.figsize"] = (14,8)

def load_data(subject):
    '''
    

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
    raw_eeg_data = fif_file.get_data()[0:64, :]
    channel_names = fif_file.ch_names[0:64]
    eeg_times = fif_file.times
    fs = fif_file.info['sfreq']
    channel_names = np.array(channel_names)
    return fif_file, raw_eeg_data, eeg_times, channel_names, fs
    



def get_eeg_epochs(fif_file, raw_eeg_data, start_time, end_time, fs):
    '''
    

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
    all_trials = all_trials[np.logical_not(np.logical_and(all_trials[:,2] > 20, all_trials[:,2] >2000))]
    target_events = all_trials[np.logical_not(np.logical_and(all_trials[:,2] > 20, all_trials[:,2] > 999))]
    
    
    event_stimulus_ids = []
    for event_index in range(len(target_events)):
        stimulus_id = target_events[event_index, 2]//10
        event_stimulus_ids.append(stimulus_id)
        
    event_start_times = all_trials[:, 0]
    for event_start_time in event_start_times:
        start_epoch = int(event_start_time) - int(start_time*fs)
        end_epoch = int(int(event_start_time) + (end_time*fs))
        epoch_data = raw_eeg_data[:, start_epoch:end_epoch]
        eeg_epochs = np.append(eeg_epochs, epoch_data)
    eeg_epochs = np.reshape(eeg_epochs, [len(all_trials), np.size(raw_eeg_data, axis=0), int(end_time*fs)])
    epoch_times = np.arange(0, np.size(eeg_epochs, axis=2))
    return eeg_epochs, epoch_times, target_events, all_trials


def get_event_truth_labels(all_trials):
    '''
    

    Parameters
    ----------
    all_trials : array of size (all trials, (event onset, post-experiment feedback, stimulus/condiiton id))
        array containing the information on all events.

    Returns
    -------
    is_target_event: boolean array
        boolean array containing labels denoting weather trial contained a target event or not.

    '''
    is_target_event = np.array([])
    for trial_index in range(len(all_trials)):
        if all_trials[trial_index, 2] < 1000 :
            is_target_event = np.append(is_target_event,True)
        elif all_trials[trial_index, 2] < 2000 :
            is_target_event = np.append(is_target_event,False)
    is_target_event = np.array(is_target_event, dtype='bool')
    return is_target_event


# def get_tempo_labels(event_stimulus_ids):
#     tempo_labels=[]
#     for stimulus_id in event_stimulus_ids:
#         if stimulus_id == 1 or stimulus_id==11:
#             tempo=212
#         elif stimulus_id == 2 or stimulus_id==12:
#             tempo=189
#         elif stimulus_id == 3 or stimulus_id==13:
#             tempo=200
#         elif stimulus_id == 4 or stimulus_id==14:
#             tempo=160
#         elif stimulus_id == 21:
#             tempo = 178
#         elif stimulus_id == 22:
#             tempo = 166
#         elif stimulus_id == 23:
#             tempo = 104
#         elif stimulus_id == 24:
#             tempo = 140
        
#         tempo_labels.append(tempo)
#     return np.array(tempo_labels)



# def get_truth_labels(tempo_labels):

#     is_trial_greater_than_170bpm = [tempo_labels[:] >= 170]
#     return is_trial_greater_than_170bpm[0]





def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_target_event, channels_to_plot, channel_names):
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

    

def perform_ICA(raw_fif_file, channel_names, top_n_components):
    '''
    

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
    None.

    '''
    picks_eeg = mne.pick_types(raw_fif_file.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')[0:64]
    ica = mne.preprocessing.ICA(n_components=64, random_state=97, max_iter=800)
    # picks = mne.pick_types(raw_fif_file.info, meg=False, eeg=True, eog=False, stim=False)
    ica.fit(raw_fif_file, picks=picks_eeg, decim=3, reject=dict(mag=4e-12, grad=4000e-13))
    mixing_matrix = ica.mixing_matrix_
    ica.plot_components(picks = np.arange(0,top_n_components))
    plt.savefig(f'figures/Top{top_n_components}ICA.png')



# %%
