import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import peakutils
import scipy.signal as _signal


#perfrom detrending using sliding window 
def moving_detrending(sig_input,sampling_rate=64):
    sig=np.copy(sig_input)
    window_size = 1*sampling_rate
    window_count = 1
    start = (window_count-1)*window_size
    end = window_count*window_size
    while(end<=len(sig)):
        if((len(sig)-end)<window_size):
            end = end + window_size
        sig[start:end] = _signal.detrend(sig[start:end])
        window_count = window_count+1
        start = (window_count-1)*window_size
        end = window_count*window_size
    return sig

def detect_RR(sig,thres,sampling_rate):
    peak_indx = peakutils.indexes(sig, thres=thres, min_dist=sampling_rate/2.5)
    time=np.arange(len(sig))
    tmp= time[peak_indx]
    timings1 = tmp[0:]
    timings = tmp[1:]
    RR_intervals = timings-timings1[:len(timings1)-1]
     
    return RR_intervals/sampling_rate,timings/sampling_rate,peak_indx

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = _signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff=5, fs=64, order=3):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = _signal.lfilter(b, a, data)
    return pd.Series(y[1000:])

#remove rr intervals that are too far from the median(rr) - probably due to noise
def medianFilter_merged(rr):
    percentageBorder = 0.8
    median = np.median(rr)
    f_rr = []
    f_time=[]
    f_rr = rr[(rr/median>=percentageBorder)  & (rr/median<=(2-percentageBorder))]
    current_time=0.0
    for rr in f_rr:
        current_time = current_time+rr
        f_time.append(current_time)
    f_time = np.array(f_time)   
    return f_rr,f_time


#perform moving average
def moving_average(sample,ma_size = 10):
    sample = pd.Series(sample)
    sample_ma = sample.rolling(ma_size).mean()
    sample_ma = sample_ma.iloc[ma_size:].values
    return sample_ma

#extract HRV features in time domain
def HRV_time(RR_intervals,print_flag):
    mean_hr = 60/np.mean(RR_intervals)
    ibi = np.mean(RR_intervals) #Take the mean of RR_list to get the mean Inter Beat Interval
    sdnn = np.std(RR_intervals) #Take standard deviation of all R-R intervals
    #find successive/neighbouring RRs (i.e., filter noise)
    RR_diff =[]
    RR_sqdiff = []
    for i in range(len(RR_intervals)-1):
        RR_diff.append(np.absolute(RR_intervals[i+1]-RR_intervals[i]))
        RR_sqdiff.append(np.power(np.absolute(RR_intervals[i+1]-RR_intervals[i]),2))  
    RR_diff=np.array(RR_diff)
    RR_sqdiff = np.array(RR_sqdiff)
    sdsd = np.std(RR_diff) #Take standard deviation of the differences between all subsequent R-R intervals
    rmssd = np.sqrt(np.mean(RR_sqdiff)) #Take root of the mean of the list of squared differences
    nn20 = [x for x in RR_diff if (x>0.02)] #First create a list of all values over 20, 50
    nn50 = [x for x in RR_diff if (x>0.05)]
    pnn20 = 100*float(len(nn20)) / float(len(RR_diff)) #Calculate the proportion of NN20, NN50 intervals to all intervals
    pnn50 = 100*float(len(nn50)) / float(len(RR_diff)) 
    sd1 = np.sqrt(0.5 * sdnn * sdnn)
    sd2 = -1
    tmp = 2.0 * sdsd * sdsd - 0.5 * sdnn * sdnn
    if tmp>0: #avoid sqrt of negative values
        sd2 = np.sqrt(2.0 * sdsd * sdsd - 0.5 * sdnn * sdnn)
    
    if(print_flag):
        print("menHR:", mean_hr)
        print("IBI:", ibi)
        print("SDNN:", sdnn)
        print("sdsd",sdsd)
        print("RMSSD:", rmssd)
        print("pNN20:",pnn20)
        print("pNN50:", pnn50)
        print("sd1:", sd1)
        print("sd2:", sd2)
        print("sd1/sd2:", sd1/sd2)
    
    return(mean_hr,ibi,sdnn,sdsd,rmssd,pnn20,pnn50,sd1,sd2,sd1/sd2)

def winsorize_signal(sample,winsorize_value):
    
    p_min = np.percentile(sample,winsorize_value)
    p_max = np.percentile(sample,100-winsorize_value)
    
    sample[sample>p_max]=p_max
    sample[sample<p_min]=p_min
    
    return sample

#https://www.mathworks.com/help/signal/ref/hampel.html
#compute median and standard deviation 
#of a window composed of the sample and its six surrounding samples
#If a sample differs from the median by more than three standard deviations, 
#it is replaced with the median. 
#reutn fistered RRs and outlier indices 
def hampel_filtering(sample_rr):
    outlier_indicies = []
    filtered_rr = []
    for i in range(len(sample_rr)):
        start = i-3
        end = i+3
        if start<0: #for the first 3 samples calculate median and std using the closest 6 samples
            start=0
            end = end+3-i
        if end>len(sample_rr)-1: #for the last 3 samples calculate median and std using the first 6 samples
            start = len(sample_rr)-7
            end=len(sample_rr)-1

        sample_med = np.median(sample_rr[start:end])
        sample_std = np.std(sample_rr[start:end])
        if abs(sample_rr[i]-sample_med)>3*sample_std:
            outlier_indicies.append(i)
            filtered_rr.append(sample_med)
#             print('outlier')
        filtered_rr.append(sample_rr[i])
    return np.array(filtered_rr),outlier_indicies
    
#filter signala and calculate HRV features in time and in frequency domain
def get_HRV_features(_sample,ma=False,detrend=False,m_deternd=False, low_pass=False,thres=.75,rr_median_filter=True,winsorize=True,dynamic_threshold=True,
                     dynamic_threshold_value=1.5,winsorize_value=25,hampel_fiter=True,sampling=1000,plt_flag=False):
    sample = _sample.copy()
  
    if low_pass: #lowpass filter
        sample = butter_lowpass_filter(sample)
    if m_deternd: #moving detrending 
        sample = moving_detrending(sample,sampling)
    if detrend:  #overall detrending
        sample= _signal.detrend(sample)
    if ma: #moving average
        sample = moving_average(sample)
        
    #winsorize the signal
    if winsorize:
        sample=winsorize_signal(sample,winsorize_value)
    if dynamic_threshold: #find the median of the min-max normalized signal
        thres = dynamic_threshold_value*np.median((sample - sample.min())/(sample.max() - sample.min()))
        
    rr,timings,peak_indx =  detect_RR(sample,thres,sampling)
    
    if hampel_fiter:
        rr,outlier_indeces = hampel_filtering(rr)
        
    timings,rr = medianFilter(timings,rr)

    if len(rr)<len(sample)/(2*sampling): #check whether HR is>30
        if(plt_flag):
            print("Bad signal. Too little RRs detected.")
        return np.array([-1]*len(feature_names_time)),sample,rr,timings,peak_indx
    elif len(rr)>len(sample)/(sampling/4):  #check whether HR is<240
        if(plt_flag):
            print("Bad signal. Too little RRs detected.")
        return np.array([-1]*len(feature_names_time)),sample,rr,timings,peak_indx
    
    hrv_time_features = HRV_time(rr,print_flag= False)

    return np.array(hrv_time_features),sample,rr,timings,peak_indx

def medianFilter(time,rr):
    percentageBorder = 0.8
    median = np.median(rr)
    idx = (rr/median>=percentageBorder)  & (rr/median<=(2-percentageBorder))
#     f_rr = rr[(rr/median>=percentageBorder)  & (rr/median<=(2-percentageBorder))]
    f_rr = np.copy(rr)
#     f_rr[~idx]=median
    f_rr = f_rr[idx]
    f_time = timestamps_from_RR(f_rr)
#     f_time = time[(rr/median>=percentageBorder)  & (rr/median<=(2-percentageBorder))]
    return f_time,f_rr


def timestamps_from_RR(rr_intervals):
    time=[]
    current_time = 0.0
    for rr in rr_intervals:
        current_time = current_time+rr
        time.append(current_time)
    return np.array(time)
feature_names_time = ['mean_hr','ibi','sdnn','sdsd','rmssd','pnn20','pnn50','sd','sd2','sd1/sd2']
feature_names_freq = ['vlf','lf','hf','lf/hf']
feature_names_freq_interpolated = ['lf_i','hf_i','lf/hf_i']

feature_names_all = np.concatenate((feature_names_time,feature_names_freq,feature_names_freq_interpolated))
