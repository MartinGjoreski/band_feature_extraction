
# coding: utf-8


import numpy as np
import peakutils
import matplotlib.pyplot as plt
import scipy.signal as signal
from random import randint
import biosppy.signals.tools as st



def filter_FIR(signal,sampling_rate,plt_flag=True,):
    filtered = st.filter_signal(signal=signal,
                 ftype="FIR",
                 band="bandpass",
                 frequency=(0.01,1),
                 order=20,
                 sampling_rate=sampling_rate)

    signal_f = filtered['signal']
    if(plt_flag):
        plt.plot(signal,label='raw',c="blue")
        plt.plot(signal_f,label='filtered',c="red")
        plt.xlabel("Sample")
        plt.ylabel("GSR value")
        plt.legend()
        plt.show()
        
    return signal_f


# In[6]:

def find_peaks(signal, sampling_rate,plt_flag=True):
    tonic = peakutils.baseline(signal, 10)
    singal_bf = signal-tonic
    indexes = peakutils.indexes(singal_bf, thres=0.3, min_dist=sampling_rate)
    if(plt_flag):
        plt.figure(figsize=(30,3))
        plt.plot(singal_bf, alpha=0.5, color='blue')
        plt.scatter(indexes, singal_bf[indexes], color='red') #Plot detected peaks
        plt.title("GSR with removed tonic")
        plt.show()
        plt.figure(figsize=(30,3))
        plt.plot(signal, alpha=0.5, color='blue', label="GSR signal")
        plt.scatter(indexes, signal[indexes], color='red') #Plot detected peaks
        plt.plot(tonic, alpha=0.5, color='green',label="GSR tonic driver")
        plt.legend()
        plt.show()
    return indexes,tonic

def find_peaks_heght_filter(signal, sampling_rate,height_threshold,plt_flag=True):
    tonic = peakutils.baseline(signal, 5)
    singal_bf = signal-tonic
    indexes = peakutils.indexes(singal_bf, thres=0.1, min_dist=sampling_rate)
    
    all_indexes = np.copy(indexes)
    good_hights = []
    bad_indexes = []
    good_hights = np.argwhere(singal_bf[indexes]> height_threshold)
    bad_hights = np.argwhere(singal_bf[indexes]<= height_threshold)
    if(len(good_hights)>0):
        indexes = np.concatenate(indexes[good_hights])
    else:
        indexes = [] #all are bad
    if(len(bad_hights)>0):
        bad_indexes = np.concatenate(all_indexes[bad_hights])
#     print(singal_bf[indexes])
#     print(sampling_rate)
    if(plt_flag):
        plt.figure(figsize=(30,3))
        plt.plot(singal_bf, alpha=0.5, color='blue',label='GSR-tonic')
        plt.scatter(all_indexes, singal_bf[all_indexes], color='red',label = 'Detected peaks') #Plot detected peaks
        plt.legend()
        plt.show()
        plt.figure(figsize=(30,3))
        plt.plot(signal, alpha=0.5, color='blue', label="GSR signal") 
        plt.scatter(indexes, signal[indexes], color='red',label = 'Good Detected peaks')
        plt.scatter(bad_indexes, signal[bad_indexes], color='purple',label = 'Bad detected peaks')
        plt.plot(tonic, alpha=0.5, color='green',label="GSR tonic driver") 
        plt.legend()
        plt.show()
        
    return indexes,tonic


import pandas as pd
def find_peaks_sliding(sig, sampling_rate,height_threshold=.1,plt_flag=True):
    window_size = 60*sampling_rate
    window_count = 1

    #detrending using sliding window. For signals in which the trend is not linear
    singal_bf = np.copy(sig)
    tonic_sliding = []
    while((window_count*window_size)<=len(sig)):
        start = (window_count-1)*window_size
        end = window_count*window_size
        if((len(singal_bf)-end)<window_size):
            end = end + window_size
        tonic_sliding.extend(peakutils.baseline(sig[start:end], 3))
        window_count = window_count+1
    sig_df = pd.DataFrame(tonic_sliding)
    tonic_sliding = sig_df.iloc[:,0].rolling(window=(3*sampling_rate),center=True).mean().values
    tonic_sliding[np.isnan(tonic_sliding)] =np.reshape(sig_df[np.isnan(tonic_sliding)].values,len(sig_df[np.isnan(tonic_sliding)].values))
    tonic_sliding = np.reshape(tonic_sliding,len(tonic_sliding))
    
    tonic = peakutils.baseline(sig, 3)

    if(len(tonic_sliding)>0):
        singal_bf = singal_bf-tonic_sliding
    else:
        singal_bf = singal_bf-tonic
    indexes = peakutils.indexes(singal_bf, thres=0.3, min_dist=sampling_rate)
    all_indexes = np.copy(indexes)
    good_hights = []
    bad_indexes = []
    good_hights = np.argwhere(singal_bf[indexes]> height_threshold)
    bad_hights = np.argwhere(singal_bf[indexes]<= height_threshold)
    if(len(good_hights)>0):
        indexes = np.concatenate(indexes[good_hights])
    if(len(bad_hights)>0):
        bad_indexes = np.concatenate(all_indexes[bad_hights])

    if(plt_flag):
        plt.figure(figsize=(30,3))
        plt.plot(singal_bf, alpha=0.5, color='blue')
        plt.scatter(indexes, singal_bf[indexes], color='red') #Plot detected peaks
        plt.title("GSR with removed tonic")
        plt.show()
        plt.figure(figsize=(30,3))
        plt.plot(sig, alpha=0.5, color='blue', label="GSR signal") 
        plt.scatter(indexes, sig[indexes], color='red')
        plt.scatter(bad_indexes, sig[bad_indexes], color='yellow')
        plt.plot(tonic, alpha=0.5, color='green',label="GSR tonic driver") #Plot semi-transparent HR
        plt.plot(tonic_sliding, alpha=0.5, color='purple',label="GSR tonic driver - sliding") #Plot semi-transparent HR
        plt.legend()
        plt.show()
    return indexes,tonic


# In[6]:

def calculate_GSR_features(signal, peaks, tonic,sampling_rate):
    mean = np.mean(signal)

    std = np.std(signal)
    q25 = np.percentile(signal,0.25)
    q75 = np.percentile(signal,0.75)
    qd = q75-q25
    deriv = np.sum(np.gradient(signal))
    power = np.mean(signal*signal)
    #peaks features
    num_peaks = len(peaks)
    rate_peaks = len(peaks)/(len(signal)/sampling_rate)
    power_peaks = np.mean(signal[peaks])
    
    derivative = np.gradient(signal)
    pos_idx = np.where(derivative > 0)[0]
    sum_pos_deriv = np.sum(derivative[pos_idx])/len(derivative)
    prop_pos_deriv = len(pos_idx)/len(derivative)
    
    deriv_tonic = np.sum(np.gradient(tonic))
    sig_tonic_difference = np.mean(signal-tonic)

    return (mean,std,q25,q75,qd,deriv,power,num_peaks,rate_peaks,power_peaks,sum_pos_deriv,prop_pos_deriv,deriv_tonic,sig_tonic_difference)


# In[7]:

def get_GSR_features(signal, sampling_rate, height_threshold, plt_flag=True):
#     signal_f =filter_FIR(signal,sampling_rate,plt_flag)
#     signal_f = mean_filter(signal,3*sampling_rate,1,sampling_rate,plt_flag)
    signal_f = signal
    peaks,tonic = find_peaks_heght_filter(signal_f,sampling_rate,height_threshold,plt_flag)

    feats = calculate_GSR_features(signal_f, peaks, tonic,sampling_rate)
#     freq_feats = GSR_freq(signal_f,sampling_rate,plt_flag,print_flag=plt_flag)
    
    peaks,ends,starts=get_peak_intervals(signal_f,peaks,sampling_rate,plt_flag)
    peak_features =get_peak_intervals_features(signal_f,peaks,starts,ends,sampling_rate)
    significant_change_features = significant_change(signal,sampling_rate,plt_flag,plt_flag)
#     print('significant_change_features',significant_change_features)

#     return np.concatenate((feats,freq_feats,peak_features,significant_change_features))
    return np.concatenate((feats,peak_features,significant_change_features))


def get_GSR_features_old(signal, sampling_rate, plt_flag=True):
    signal_f =filter_FIR(signal,sampling_rate,plt_flag)
    peaks,tonic = find_peaks(signal_f,sampling_rate,plt_flag)
    feats = calculate_GSR_features(signal_f, peaks, tonic,sampling_rate)
#     freq_feats = GSR_freq(signal_f,sampling_rate,plt_flag,print_flag=plt_flag)

    return feats


def GSR_freq(s,fs,plot_flag,print_flag):   
    nfft=16
    ff, Pxx_spec = signal.periodogram(s, fs,nfft=nfft,detrend=None)
#     print(fs)
    if(plot_flag):
        plt.plot(s,label="Signal freq")
        plt.legend()
        plt.show()
        plt.semilogy(ff, Pxx_spec)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.xlim(0,fs//2)
        plt.show()
    # get the power in the band [0-0.5]
    current_f=0.0 
    increment = 0.1
    feats=[]
    while(current_f<0.6):
        feat =  np.trapz(abs(Pxx_spec[(ff>=current_f) & (ff<=current_f+increment)]))
        feats.append(feat)
#         if(print_flag):
#             print(current_f,"-",current_f+increment, feat)
        current_f = current_f+increment
    return feats


def significant_increase(sig, fs,print_flag):
    #5 seconds
    win_size = 5*fs
    sig_change_threshold = 1.05 #5%
    sig_counter=0
    sig_duration_threshold = 15*fs #10% change should be sustained for a duration of 15 seconds
    sig_duration = 0
    sig_windows = []
    sig_windows_all = []

    for idx in range(len(sig)//win_size-1):
#         print('inside')
        win_prev = sig[idx*win_size]
        win_next = sig[(idx+1)*win_size]
        if win_prev==0:
            win_prev = win_prev+0.00001
        if(win_next/win_prev)>sig_change_threshold:
            sig_counter = sig_counter+1
#             print("Sig increase")
            sig_windows.append(win_prev)
        else:
            if sig_counter*win_size>=sig_duration_threshold: #foe how manu windows there was a sig change?
                sig_duration = sig_duration + (sig_counter*win_size)
                if(print_flag):
                    print("Significant increase ended")
                sig_windows_all.extend(sig_windows)
            sig_counter = 0
            sig_windows = []
#         if(print_flag):
#             print(idx*win_size,(idx+1)*win_size,win_next/win_prev)
    if(sig_counter*win_size>=sig_duration_threshold):
        sig_duration = sig_duration + (sig_counter*win_size)
        
    #how many seconds there has been a significant increase
    mean = 0
    intensity = 0
    change = 0
    speed= 0
    if len(sig_windows_all)>0:
        mean = np.mean(sig_windows_all)
        intensity = np.mean(sig_windows_all)*sig_duration
        change = max(sig_windows_all) - min(sig_windows_all) 
        speed = change/sig_duration
    return (sig_duration,mean,intensity,change,speed)

def significant_decrease(sig, fs, print_flag):
    #5 seconds
    win_size = 5*fs
    sig_change_threshold = 1.05
    sig_counter=0
    sig_duration_threshold = 15*fs #10% change should be sustained for a duration of 15 seconds
    sig_duration = 0
    
    sig_windows = []
    sig_windows_all = []
    
    for idx in range(len(sig)//win_size-1):
        win_prev = sig[idx*win_size]
        win_next = sig[(idx+1)*win_size]
        if win_next==0:
            win_next = win_prev+0.00001
        if (win_prev/win_next)>sig_change_threshold:
            sig_counter = sig_counter+1
            sig_windows.append(win_prev)
        else:
            if (sig_counter*win_size)>=sig_duration_threshold:
                sig_duration = sig_duration + (sig_counter*win_size)
                if(print_flag):
                    print("Significant decrease ended")
                sig_windows_all.extend(sig_windows)
            sig_counter = 0
            sig_windows = []
#         if(print_flag):
#             print(idx*win_size,(idx+1)*win_size,win_prev/win_next)
    if(sig_counter*win_size>=sig_duration_threshold):
        sig_duration = sig_duration + (sig_counter*win_size)
        
    #how many seconds there has been a significant decrease
    mean = 0
    intensity = 0
    change=0
    speed = 0
    if len(sig_windows_all)>0:
        mean = np.mean(sig_windows_all)
        intensity = np.mean(sig_windows_all)*sig_duration
        change =  min(sig_windows_all) -max(sig_windows_all) 
        speed = change/sig_duration
    return (sig_duration,mean,intensity,change,speed)

def significant_change(sig, fs, plt_flag,print_flag):
    a = significant_increase(sig,fs,print_flag)
    b = significant_decrease(sig,fs,print_flag)
    
    return np.concatenate([a,b])

def get_peak_intervals(sig,peak_indexes,sampling_frequency,plt_flag):
    window_size = sampling_frequency
    window_slide=1
    inertion=.001
    ends = []
    starts = []
    for start_idx in peak_indexes:
        
        #go backwards
        mean_prev = np.mean(sig[start_idx:(start_idx+window_size)])
        window_start = start_idx-window_size
        window_end = start_idx
        mean_current =  np.mean(sig[window_start:window_end])
        while(window_start>=0 and (mean_current+inertion)<=mean_prev):
            window_end = window_end-window_slide
            window_start = window_start-window_slide
            mean_prev=mean_current
            mean_current =  np.mean(sig[window_start:window_end])
        if(window_end<0):
            window_end = 0
        value = window_end
        if(value>start_idx):
            value=start_idx-window_size
        if(value<0):
            value=0
        starts.append(value)

        #go forward
        mean_prev = np.mean(sig[start_idx:(start_idx+window_size)])
        window_start = start_idx+window_slide
        window_end = window_start+window_size
        mean_current =  np.mean(sig[window_start:window_end])
        while(window_end<=len(sig) and (mean_current+inertion)<=mean_prev):
            window_start = window_start+window_slide
            window_end = window_end+window_slide
            mean_prev=mean_current
            mean_current =  np.mean(sig[window_start:window_end])
        if(window_start>=len(sig)):
            window_start = len(sig)-1
            
        value = window_start
        if(value<=start_idx):
            value=start_idx+window_size
        if(value>=len(sig)):
            value = len(sig)-1
            
        ends.append(value)

    #filter bad-short peaks
    inc_duration_threshold = 1
    dec_duration_threshold = 1
    inc_amplitude_threshold  = .1
    dec_amplitude_threshold  = .1
    good_indexes = []
    bad_indexes = []
    for i in range(len(peak_indexes)):
        inc_duration = (peak_indexes[i]-starts[i])/sampling_frequency
        dec_duration = (ends[i]-peak_indexes[i])/sampling_frequency
        inc_amplitude = (sig[peak_indexes[i]]-sig[starts[i]])
        dec_amplitude = (sig[peak_indexes[i]]-sig[ends[i]])
#         print(i,inc_duration,dec_duration,inc_amplitude,dec_amplitude)
        if (inc_duration>=inc_duration_threshold and
            dec_duration>=dec_duration_threshold and
            inc_amplitude>=inc_amplitude_threshold and 
            dec_amplitude>=dec_amplitude_threshold):
            good_indexes.append(i)
        else:
            bad_indexes.append(i)
    peak_indexes = np.array(peak_indexes)       
    bad_peak_indexes = peak_indexes[bad_indexes]
    peak_indexes = peak_indexes[good_indexes]
    ends = np.array(ends)
    starts = np.array(starts)
    ends= ends[good_indexes]
    starts= starts[good_indexes]
#     print('peak_indexes:',peak_indexes)
    if(plt_flag and len(peak_indexes)>0):
        plt.figure(figsize=(30,3))
        plt.plot(sig,label='GSR')
        plt.scatter(peak_indexes, sig[peak_indexes], color='red',label='Good Detected peaks') #Plot detected peaks
        plt.scatter(bad_peak_indexes, sig[bad_peak_indexes], color='purple',label='Bad Detected peaks') #Plot detected peaks

        plt.scatter(ends, .001+sig[ends], color='orange',label='Peak end') #Plot detected peaks
        plt.scatter(starts, .001+sig[starts], color='green',label='Peak start') #Plot detected peaks
        plt.legend()
        plt.show()

    return peak_indexes,np.array(ends),np.array(starts)

def get_peak_intervals_features(sig,peak_indexes,starts,ends,sampling_frequency):
    if(len(peak_indexes)>0):
        max_peak_idx  = np.argmax(sig[peak_indexes])
        
        max_peak_start = starts[max_peak_idx]
        max_peak_end = ends[max_peak_idx]

        max_peak_amlitude_change_before = sig[peak_indexes[max_peak_idx]]-sig[max_peak_start]
        max_peak_amlitude_change_after = sig[peak_indexes[max_peak_idx]]-sig[max_peak_end]

        max_peak_change_ratio = max_peak_amlitude_change_before/max_peak_amlitude_change_after

        avg_peak_amlitude_change_before = np.median(sig[peak_indexes]-sig[starts])
        avg_peak_amlitude_change_after = np.median(sig[peak_indexes]-sig[ends])
        
        avg_peak_change_ratio=0
        if avg_peak_amlitude_change_after!=0:
            avg_peak_change_ratio = avg_peak_amlitude_change_before/avg_peak_amlitude_change_after

        max_peak_increase_time = (peak_indexes[max_peak_idx]-max_peak_start)/sampling_frequency
        max_peak_decrease_time = (max_peak_end-peak_indexes[max_peak_idx])/sampling_frequency
        
        max_peak_duration = (max_peak_end-max_peak_start)/sampling_frequency

        max_peak_change_ratio=0
        if max_peak_decrease_time!=0:
            max_peak_change_ratio = max_peak_increase_time/max_peak_decrease_time
            
        avg_peak_increase_time = np.mean(peak_indexes-starts)/sampling_frequency
        avg_peak_decrease_time = np.mean(ends-peak_indexes)/sampling_frequency
        avg_peak_duration = np.mean(ends-starts)*sampling_frequency
        avg_peak_change_ratio=0
        if(avg_peak_decrease_time!=0):
            avg_peak_change_ratio = avg_peak_increase_time/avg_peak_decrease_time
            

        max_peak_response_slope_before = np.mean(np.diff(sig[max_peak_start:peak_indexes[max_peak_idx]]))
        if np.isnan(max_peak_response_slope_before):
            max_peak_response_slope_before = 0

        max_peak_response_slope_after = np.mean(np.diff(sig[peak_indexes[max_peak_idx]:max_peak_end]))
        if np.isnan(max_peak_response_slope_after):
            max_peak_response_slope_after = 0
            
           

        signal_overall_change = np.max(sig)-np.min(sig)
        change_duration = np.abs((np.argmax(sig) - np.argmin(sig)))/sampling_frequency
        change_rate=0
        if(signal_overall_change!=0):
            change_rate = change_duration/signal_overall_change
            gsr_peak_features = [max_peak_amlitude_change_before, max_peak_amlitude_change_after,
                            avg_peak_amlitude_change_before, avg_peak_amlitude_change_after, 
                             avg_peak_change_ratio,max_peak_increase_time, max_peak_decrease_time, 
                             max_peak_duration, max_peak_change_ratio, 
                             avg_peak_increase_time,avg_peak_decrease_time,avg_peak_duration, 
                           max_peak_response_slope_before,max_peak_response_slope_after,signal_overall_change,
                            change_duration,change_rate]
    else:
        num_features = 17
        gsr_peak_features = np.array([0]*num_features)
#         print('bad features',gsr_peak_features)
    return gsr_peak_features

def mean_filter(s,windows_size,window_slide,sampling_rate,plt_flag=True):
    mean_s = []
    start=0
    end = windows_size
    while(end<=len(s)):
        mean_s.append(np.mean(s[start:end]))
        start = start + window_slide
        end = start+windows_size
    if(plt_flag):
        plt.plot(s,label='original')
        plt.plot(mean_s,label='mean_filter')
        plt.legend()
        plt.show()

    return np.array(mean_s)


feature_names_small = ['meanGsr', 'stdGsr','q25Gsr', 'q75Gsr', 'qdGsr', 'derivGsr', 'powerGsr', 'num_peaks',
       'rate_peaks', 'power_peaks','sum_pos_deriv', 'prop_pos_deriv',
       'deriv_tonic', 'sig_tonic_difference']
feature_names_medium = ['meanGsr', 'stdGsr','q25Gsr', 'q75Gsr', 'qdGsr', 'derivGsr', 'powerGsr', 'num_peaks',
       'rate_peaks', 'power_peaks', 'sum_pos_deriv','prop_pos_deriv',
       'deriv_tonic', 'sig_tonic_difference','fp01', 'fp02', 'fp03',
       'fp04', 'fp05', 'fp06']

feature_names_all = ['meanGsr', 'stdGsr','q25Gsr', 'q75Gsr', 'qdGsr', 'derivGsr', 'powerGsr', 'num_peaks',
       'rate_peaks', 'power_peaks','sum_pos_deriv','prop_pos_deriv',
       'deriv_tonic', 'sig_tonic_difference',
#        'fp01', 'fp02', 'fp03','fp04', 'fp05', 'fp06', 
       'max_peak_amlitude_change_before', 'max_peak_amlitude_change_after',
        'avg_peak_amlitude_change_before', 'avg_peak_amlitude_change_after', 
        'avg_peak_change_ratio','max_peak_increase_time', 'max_peak_decrease_time', 
        'max_peak_duration', 'max_peak_change_ratio', 
        'avg_peak_increase_time','avg_peak_decrease_time','avg_peak_duration', 
        'max_peak_response_slope_before','max_peak_response_slope_after',
        'signal_overall_change','change_duration','change_rate',
        'significant_increase_duration','significant_increase_mean', 'significant_increase_intensity', 
        'significant_increase_change', 'significant_increase_speed','significant_decrease_duration', 
        'significant_decrease_mean','significant_decrease_intensity', 'significant_decrease_change',
        'significant_decrease_speed']


        