"""
Created on Tue Nov 30 16:36:55 2021

test_Project3.py

File that calls functions load_data, get_eeg_epochs, get_truth_event_labels, plot_power_spectrum, perform_ICA that are 
defined in the Project3.py file.

@author: spenc, JJ

    
"""
#%% Import Statements
from import_ssvep_data import get_frequency_spectrum
import Project3
import numpy as np
import matplotlib.pyplot as plt

#%% Loading in the data
plt.rcParams["figure.figsize"] = (14,8)

fif_file, raw_eeg_data, eeg_times, channel_names, fs = Project3.load_data('13')

#%% Epoching the data
start_time = 0
# Explanation for end time from person behind experiemnt. A trial epoch (as I used it) starts at a trial onset marker (I used the audio onsets when available). The duration depends on what you want to do. The length of the stimuli is known but it is different for each stimulus. You can either have epochs that correspond to the full audio stimulus but vary in length (e.g. for stimulus reconstruction experiments). Or you can cut of at the length of the shortest stimulus if they need to have the same length (e.g. for stimulus recognition experiments).
end_time = 7.6

eeg_epochs, epoch_times, all_trials = Project3.get_eeg_epochs(fif_file, raw_eeg_data, start_time, end_time, fs)

#%% Extract truth labels
is_target_event = Project3.get_event_truth_labels(all_trials)

#%% Calculating and plotting mean power spectrum for specified channels
# eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)


# Project3.plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_target_event, channels_to_plot, channel_names)

#%% Computing ICA
top_n_components = 10

ica = Project3.perform_ICA(fif_file, channel_names, top_n_components)

# %%
# components = 6
components = np.arange(0, top_n_components, 1)
source_activations = Project3.plot_component_variance(ica, components, eeg_epochs, is_target_event)


component = 2
predicted_labels = Project3.make_prediction(source_activations, component, is_target_event)

truth_labels_binary = is_target_event*1
accuracy = Project3.evaluate_predictions(predicted_labels, truth_labels_binary)

