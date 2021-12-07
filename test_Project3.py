from filter_ssvep_data import make_bandpass_filter
from filter_ssvep_data import filter_data


fif_file, raw_eeg_data, eeg_times, channel_names, fs = load_data('13')

start_time = 0
end_time = 5

filter_coefficients = make_bandpass_filter(40, 60, 1000, fs, filter_type = 'hann')

filtered_data = filter_data(raw_eeg_data, filter_coefficients)

eeg_epochs, epoch_times, target_events, all_trials, event_stimulus_ids = get_eeg_epochs(fif_file, filtered_data, start_time, end_time, fs)


tempo_labels = get_tempo_labels(event_stimulus_ids)


is_trial_greater_than_170bpm=get_truth_labels(tempo_labels)


eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)


channels_to_plot = ['TP7']