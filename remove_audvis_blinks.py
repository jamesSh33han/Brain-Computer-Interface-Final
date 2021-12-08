# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 12:01:19 2021

Lab 5: Spatial Components

Defines functions to load the AudVisData and plot electrodes Fpz, Cz, and Iz, plot each component from the data in a separate
subplot, get the source activation timecourses from the EEG data, take a list of sources the user would like removed from the 
data, zero out the specified sources, and transform the results back into the electrode space. The final function in this module
plots each of the EEG timecourses (raw, reconstructed, cleaned) on each subplot.

@author: spencer, JJ
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_topo

#%% Part 1: Load the Data
def load_data(data_directory, channels_to_plot):
    '''
    loads the raw EEG data, plots data from each specified electrode channel in channels_to_plot

    Parameters
    ----------
     data_directory : string
        path to folder that contains EEG data 
    channels_to_plot : list
        specifies the names of the electrode channels that you with to plot the raw data for

    Returns
    -------
    data : dict
        dictionary containing the raw EEG data that was loaded in from the specified data_directory

    '''
    data = np.load(data_directory+'AudVisData.npy', allow_pickle=True)[()]
    # if the user enters no channels to plot pass on to just return data
    if len(channels_to_plot) == 0:
        pass 
    else:
        fig, ax = plt.subplots(nrows = len(channels_to_plot), ncols = 1, sharex=True)
        for channel_index, channel in enumerate(channels_to_plot):
            ax[0].set_title('Raw AudVis EEG Data')
            # get index of channel to plot
            index_to_plot = np.where(data['channels']==channel)[0][0]
            times = np.arange(0, np.size(data['eeg'], axis=1)/(data['fs']), step=1/(data['fs']))
            ax[channel_index].plot(times,data['eeg'][index_to_plot], label = f'{channel}')
            ax[channel_index].set_xlabel('Time (s)')
            ax[channel_index].set_ylabel(f'Voltage on {channel} (µV)')
        plt.tight_layout()
    return data

#%% Part 2: Plot the Components
def plot_components(mixing_matrix, channels, components_to_plot=np.arange(0,10,1)):
    '''
    Plots each specified component from the list in a seperate subplot

    Parameters
    ----------
    mixing_matrix : array of float of size (channels, channels)
        The mixing matrix represents the ICA components
    channels : array of string
        The channel names
    components_to_plot : array of int, optional
        List of components to plot. The default is np.arange(0,10,1), which is components 0-9

    Returns
    -------
    None.

    '''
    plt.figure('Topographical')
    # for each component in the user's entered array
    for component_index, component in enumerate(components_to_plot):
        plt.subplot(2,5,component_index+1)
        im, cbar = plot_topo.plot_topo(list(channels), mixing_matrix[:,component], title=f'ICA component {component}', cbar_label ="", montage_name = 'standard_1005')
    
    plt.tight_layout()

#%% Part 3: Transform into Source Space
def get_sources(eeg, unmixing_matrix, fs, sources_to_plot):
    '''
    Returns source activation timecourses from eeg data and plots results for specified sources

    Parameters
    ----------
    eeg : array of float size of (channels, samples)
        raw EEG data from the dictionary 'data'
    unmixing_matrix : array of float of size (channels, channels)
        Unmixing matrix - ICA source tranformation to get data into source space
    fs : float
        The sampling rate
    sources_to_plot : list
        List of specified sources to plot

    Returns
    -------
    source_activations : array of float of size (channels, samples)
        Source activation timecourses

    '''
    # get source activations by multiplying matrices
    source_activations = np.matmul(unmixing_matrix, eeg)
    # check if sources to plot is 0
    if len(sources_to_plot) == 0:
        pass
    else:
        fig, ax = plt.subplots(nrows = len(sources_to_plot), ncols = 1, sharex=True)
        for source_index, source in enumerate(sources_to_plot):
            ax[0].set_title('AudVis Data in ICA Source Space')
            # ax[channel_index].subplot(len(channels_to_plot), 1, channel_index+1)
            times = np.arange(0, np.size(eeg, axis=1)/(fs), step=1/(fs))
            ax[source_index].plot(times, source_activations[source], label = 'reconstructed')
            ax[source_index].set_xlabel('Time (s)')
            ax[source_index].set_ylabel(f'Source {source} (uV)')
            ax[source_index].legend()
        plt.tight_layout()
        

    return source_activations

#%% Part 4: Remove Artifact Components
def remove_sources(source_activations, mixing_matrix, sources_to_remove):
    '''
    Zero's out the specified sources_to_remove, transforms the results back to electrode space, and returns the cleaned data

    Parameters
    ----------
    source_activations : array of float of size (channels, samples)
        Source activation timecourses
    mixing_matrix : array of float of size (channels, channels)
        The mixing matrix represents the ICA components
    sources_to_remove : list
        List of sources that the user would like removed from the data

    Returns
    -------
    eeg_cleaned : array of float of size (channels, samples)
        Cleaned EEG data that does not contain the specified sources_to_remove

    '''
    # copy the source activations
    source_activations_altered = source_activations.copy()
    # set the source activations equal to 0 at the sources to remove
    source_activations_altered[sources_to_remove] = 0
    # get cleaned matrix by multiplying matrices
    eeg_cleaned = np.matmul(mixing_matrix, source_activations_altered)
    return eeg_cleaned

#%% Part 5: Transform Back Into Electrode Space
def compare_reconstructions(eeg, cleaned_eeg, reconstructed_eeg, fs, channels, channels_to_plot):
    '''
    Plots each of the EEG timecourses (raw, reconstructed, cleaned) in turn on each subplot

    Parameters
    ----------
    eeg : array of float
        raw EEG data from the dictionary 'data'
    cleaned_eeg : array of float of size (channels, samples)
        Cleaned EEG data that does not contain the specified sources_to_remove
    reconstructed_eeg : array of float of size channels(samples)
        Array of eeg data representing the reconstructed eeg data, goig from our cleaned eeg to our original eeg.
    fs : float
        The sampling rate
    channels : array of string
        The channel names
    channels_to_plot : list
        specifies the names of the electrode channels that you want to plot

    Returns
    -------
    None.

    '''
    fig, ax = plt.subplots(nrows = len(channels_to_plot), ncols = 1, sharex=True)
    for channel_index, channel in enumerate(channels_to_plot):
        ax[0].set_title('AudVis EEG Data Reconstructed and Cleaned after ICA')
        index_to_plot = np.where(channels==channel)[0][0]
        times = np.arange(0, np.size(eeg, axis=1)/(fs), step=1/(fs))
        ax[channel_index].plot(times,eeg[index_to_plot], label = 'eeg')
        ax[channel_index].plot(times,reconstructed_eeg[index_to_plot], linestyle='dashed', label = 'reconstructed', alpha=.7)       
        ax[channel_index].plot(times,cleaned_eeg[index_to_plot], linestyle='dotted', label='cleaned', alpha=.8)
        ax[channel_index].set_xlabel('Time (s)')
        ax[channel_index].set_ylabel(f'Voltage on {channel} (uV)')
        ax[channel_index].legend()
    plt.tight_layout()
    
