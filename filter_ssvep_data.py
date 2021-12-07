#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 18:16:48 2021

@author: Spencer, Kai
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


plt.rcParams["figure.figsize"] = (18,10)

def make_bandpass_filter(low_cutoff, high_cutoff, filter_order, fs, filter_type = 'hann'):
    '''
    This function makes a  filter of the specified type, calculates the coefficients and 
    plots the frequency response 

    Parameters
    ----------
    low_cutoff : int
        input lowest frequency of the bandpass
    high_cutoff : int
        input highest frequency of the bandpass
    filter_type : str
        input filter type, default is hann
    filter_order : int
        input the order of the filter
    fs : int
        input the sampling frequency of the filter

    Returns
    -------
    filter_coefficients : (1002,) float array
        array of the filter coefficients. 

    '''
    #Creating the filter for the impulse response
    filter_coefficients = signal.firwin(filter_order+1,[low_cutoff, high_cutoff], window = filter_type, pass_zero = False, fs = fs)
    #Creating the filter for the frequency response
    frequencies,frequency_response = signal.freqz(filter_coefficients, a = 1, fs = fs, include_nyquist=False)
    frequency_response = 10*np.log10(abs(frequency_response))
    
    plt.figure('filter')
    plt.subplot(2,1,1)
    plt.plot(frequencies,frequency_response)
    #Labels for legend
    labels = ['12Hz', '12Hz','15Hz', '15Hz']
    plt.legend(labels)
    #Plot Vertical line at 12Hz
    plt.axvline(12,linestyle = '--')
    #Plot Vertical line at 15Hz
    plt.axvline(15,color = 'orange',linestyle = '--')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.title('Hann FIR Filter Frequency Response')
    plt.grid(b = True)
    #Create delay time vector for x axis
    delay = np.arange(0,len(filter_coefficients)/fs,1/(fs))
    #Plot Impluse Response
    plt.subplot(2,1,2)
    plt.plot(delay,filter_coefficients)
    plt.xlabel('Delay [sec]')
    plt.ylabel('Gain')
    plt.title('Hann FIR Filter Impluse Response')
    # add grid
    plt.grid(b = True)
    labels2 = ['12Hz','15Hz']
    # add legend
    plt.legend(labels2)
    plt.tight_layout()
    #Save figure
    # plt.savefig(f'hann_filter_{low_cutoff}-{high_cutoff}_order{filter_order}.png')
    
    
    return filter_coefficients


def filter_data(eeg, filter_coefficients):
    '''
    This function filters the data forward and backwards with respect to time 
    with an IIR filter

    Parameters
    ----------
    data : dictionary 
        data dictionary containing the SSVEP data. EEG data is extracted using data['eeg']
    filter_coefficients : (1002,) float array
        array of filter coefficients generated using make_bandpass_filter

    Returns
    -------
    filtered_data : (32,469680) float array
        array of filtered data

    '''
    # extract eeg data to be filtered and convert to Uv
    
    # filter data
    filtered_data = signal.filtfilt(filter_coefficients, a = 1, x = eeg)
    
    return filtered_data
    

def get_envelope(data, filtered_data, frequency_to_plot, channel_to_plot = None):
    '''
    this function creates an envilope of the filtered data to plot the amplitude 
    of the signal

    Parameters
    ----------
   data : dictionary 
        data dictionary containing the SSVEP data. the fs is extracted with data['fs']
    filtered_data : (32,469680) float array
        array of filtered data
    
    frequency_to_plot : str
        input the frequency to plot
    channel_to_plot : str, optional
        inpuit the EEG channels you wish to plot. The default is None

    Returns
    -------
    envelope : (32, 469680) float array
        array of the amplitude of the filtered signal 

    '''
    # get eeg data from dictionary
    eeg = data['eeg']*1e6
    # get fs from data dictionary
    fs = data['fs']
    # get channels array
    channels = data['channels']
    #Create time vector for xaxis on plot
    time = np.arange(0,len(np.transpose(eeg))/fs,1/fs)
    #Create bool of channel to plot = given string
    channel_numbers = channels == channel_to_plot
    #Create tuple with location of row to plot
    channel_index = np.where(channel_numbers == True)[0][0]
    #Apply hilbert transform to raw data
    envelope = np.abs(signal.hilbert(filtered_data))
    # create and label figure
    plt.figure(f'{frequency_to_plot}')
    # plot filtered data
    plt.plot(time,filtered_data[channel_index])
    # plot envelope
    plt.plot(time,envelope[channel_index])
    # label
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (uV)')
    # add grid
    plt.grid(b = True)
    # titel
    plt.title(f'{frequency_to_plot}Hz BPF Data')
    labels = ['Filtered Signal','Envelope']
    # add legend
    plt.legend(labels)
    return envelope

def plot_ssvep_amplitudes(data_dict, envelope_first_freq, envelope_second_freq, channel_to_plot, first_frequency, second_frequency, subject):
    '''
    

    Parameters
    ----------
    data_dict : dictionary 
        data dictionary of SSVEP data.
    envelope_first_freq : (1002,) float array
        envelope array of the lower frequency.
    envelope_second_freq : (1002,) float array
        envelope array of higher frequency.
    channel_to_plot : str
        name of the channel to plot.
    first_frequency : str
        name of the lower frequency used to title the plot.
    second_frequency : str
        name of the higher frequency used to title the plot.
    subject : int
        number of which subject to plot.

    Returns
    -------
    None.

    '''
    # get frequency
    fs = data_dict['fs']
    fig, axs = plt.subplots(2,1,sharex = True)
    #Set grids
    axs[0].grid(b = True)
    #Loop to plot SSVEP amplitudes
    for plot_index in range(len(data_dict['event_samples'])-1):
        #Compute start time for each event
        start_time = data_dict['event_samples'][plot_index]*(1/fs)
        #Compute end time for each event
        end_time = start_time+(data_dict['event_durations'][0]/fs)
        #Plot the duration of each event at each frequency
        axs[0].plot([start_time,end_time],[data_dict['event_types'][plot_index],data_dict['event_types'][plot_index]],'-o',color = 'blue')
    # label
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Flash Frequency')
    axs[0].set_title(f'Subject {subject} SSVEP Amplitudes')
    # add grid
    plt.grid()
    
    #Create bool of channel to plot = given string
    channel_numbers = data_dict['channels'] == channel_to_plot
    #Create tuple with location of row to plot
    channel_index = np.where(channel_numbers == True)[0][0]
    
    #Create time vector for xaxis on plot
    time = np.arange(0,len(np.transpose(data_dict['eeg']))/data_dict['fs'],1/data_dict['fs'])
    
    #Plot both 12Hz and 15Hz envelopes
    axs[1].plot(time,envelope_first_freq[channel_index])
    axs[1].plot(time,envelope_second_freq[channel_index])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Voltage (uV)')
    axs[1].set_title('Envelope Comparison')
    axs[1].grid(b = True)
    # plt.show()
    labels = [f'{first_frequency} Envelope',f'{second_frequency} Envelope']
    axs[1].legend(labels)
