
import numpy as np
import matplotlib.pyplot as plt
from rpm_extract import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_conversor'))
from data_handler import *
from encoder import *





#initialize Case_1
path = r"D:\IC EMBRAER\microphones\DELFT_4000_EXPDELFT_J0_0"
Case_1 = AeroPropCase(path)

#import variables from microphone
waveform = Case_1.microphone.signals[0][:]
fs = Case_1.microphone.fs #Aquisition frequency signal
blades = 2 #number of blades
time = np.linspace(0,len(waveform)/fs,len(waveform))


#import variables from shaft and calculate fs_tacho and timedata from tacho
load_cases = AeroPropCase(r"D:\IC EMBRAER\microphones\DELFT_4000_EXPDELFT_J0_0")
encoder_data = load_cases.shaft.encoder
rpm_inst = load_cases.shaft.rpm_inst
time_tacho = load_cases.shaft.time_taco
fs_tacho = 1/(time_tacho[2]-time_tacho[1])
timedata = np.linspace(0,len(encoder_data)/fs_tacho,len(encoder_data))



#get experimental rpm from the encoder data


exp_rpm = get_rpm_from_enconder(time_tacho, encoder_data, 600)


#demodulation method test
rpm_estimate = 4000 
band = 5
demo = demodulation_method(waveform, fs, rpm_estimate, band)
demo = np.multiply(demo,60)

#spectrogram method test
band = 5
spec = spectrogram_method(waveform, fs, rpm_estimate, band)
spec = np.multiply(spec,60)


#hybrid method test
band = 5
hyb = hybrid_method(waveform, fs, rpm_estimate, band)
hyb = np.multiply(hyb,60)


#plot inst frequencies of each method

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.plot(exp_rpm)
plt.title("Experimental RPM")
plt.subplot(2,2,2)
plt.plot(demo)
plt.title("Demodulation Method")
plt.subplot(2,2,3)
plt.plot(spec)
plt.title("Spectrogram Method")
plt.subplot(2,2,4)
plt.plot(hyb)
plt.title("Hybrid Method")
plt.tight_layout()
plt.show()