#%%
from filter_ssvep_data import make_bandpass_filter
from filter_ssvep_data import filter_data
import Project3

fif_file, raw_eeg_data, eeg_times, channel_names, fs = Project3.load_data('13')

start_time = 0
end_time = 1.5



eeg_epochs, epoch_times, target_events, all_trials, event_stimulus_ids = Project3.get_eeg_epochs(fif_file, raw_eeg_data, start_time, end_time, fs)

is_target_event = Project3.get_event_truth_labels(all_trials)

tempo_labels = Project3.get_tempo_labels(event_stimulus_ids)


eeg_epochs_fft, fft_frequencies = Project3.get_frequency_spectrum(eeg_epochs, fs)


channels_to_plot = ['Fz']


Project3.plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_target_event, channels_to_plot, channel_names)

mean_eeg, rms_eeg, std_eeg = Project3.extract_eeg_features(eeg_epochs)

mixing_matrix = Project3.perform_ICA(fif_file, channel_names)

# %%
