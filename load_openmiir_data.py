# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:36:55 2021

@author: spenc
"""
import mne
import numpy as np
def load_data(subject):
    fif_file=mne.io.read_raw_fif(f'{subject}-raw.fif', preload=True)
    raw_eeg_data = fif_file.get_data()
    channel_names = fif_file.ch_names[0:64]
    eeg_times = fif_file.times
    fs = fif_file.info['sfreq']
    return fif_file, raw_eeg_data, eeg_times, channel_names, fs
    

fif_file, raw_eeg_data, eeg_times, channel_names = load_data('P01')



def get_event_types(events):
    event_types = []
    event_times = []
    for i in range(len(events)):
        event_type = ''
        event_time = events[i, 0]
        if events[i, 2] > 1000:
            events = np.delete(events, i, 0)
            # stimulus_id =  events[i, 2]// 10
            
            # event_types.append(stimulus_id)
        event_times.append(event_time)
        
    return events

def get_epochs(fif_file, events):
    epochs = mne.Epochs(fif_file, events, tmin=-0.2, tmax=0.8, baseline=(0,0)).get_data()


