#%%

'''
Some questions we have:
    
    1. 
    2.
    3.

'''



from import_ssvep_data import get_frequency_spectrum
import Project3
import numpy as np

fif_file, raw_eeg_data, eeg_times, channel_names, fs = Project3.load_data('13')

start_time = 0
end_time = 1.5


eeg_epochs, epoch_times, target_events, all_trials = Project3.get_eeg_epochs(fif_file, raw_eeg_data, start_time, end_time, fs)

is_target_event = Project3.get_event_truth_labels(all_trials)


eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)

channels_to_plot = ['Fz']

Project3.plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_target_event, channels_to_plot, channel_names)


top_n_components = 10

Project3.perform_ICA(fif_file, channel_names, top_n_components)


# %%
