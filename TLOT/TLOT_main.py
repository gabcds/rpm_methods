
from scipy import signal, integrate
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_conversor'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'LVA_techniques'))
from data_handler import *
from rpm_extract import *
from scipy.signal import chirp, czt
from numpy.fft import fft, ifft, fftshift



# (1) Finding the IF (intantaneous frequency) of the raw signal
def central_freq(x, fs, rpm_estimate, band):
    return hybrid_method(x, fs, rpm_estimate, band)


def get_IP_MNI(f_max):


    """Obtain the corrected instantaneus phase (theta_jj) by MNI
    Args:
        f_max (array_like): Inst. frequency matrix
        fs (float): sampling frequency [Hz]

    Returns: 
        inst_phase_cor (array_like): Corrected inst. phase matrix
        
    """ 
    time = np.linspace(0,len(f_max),len(f_max))  
    interp_f_max = interp1d(time,f_max,kind="linear",fill_value="extrapolate")
    tn = time[2]-time[1]
    n_iter = len(f_max)/tn
    i = 0
    a = tn
    b = a+tn
    inst_phase_cor=[tn*(interp_f_max(0)+interp_f_max(tn))/4] #initialize with CNI
    
    while i<n_iter:
        f_romb = integrate.romberg(interp_f_max,a,b)
        inst_phase_cor.append(f_romb)
        a += tn
        b += tn
        i += 1
    return np.array(inst_phase_cor)


def angular_resampling(f_max,phase,signal):

    """Perform a angular resampling based on 3 consecutive points according to Taylor polynomial

    Args:
        
        phase (array_like): instant phase calculated according to MNI
        signal (array_like): time domain signal
    
    Returns:
        signal resampled function
    """
    time = np.linspace(0,len(signal),len(signal))
    time_taco = np.linspace(0,len(f_max),len(f_max)) 
    num_zeros_phase = (3 - (len(phase) % 3)) % 3
    num_zeros_tt = (3 - (len(phase) % 3)) % 3
    M = len(phase)+1 #used later for reshaping 
    phase = np.append(phase, np.zeros(num_zeros_phase))
    time_taco = np.append(time_taco, np.zeros(num_zeros_tt))
    N = len(phase)//3
    time_taco = np.reshape(time_taco,(N,3)) #transforming 1D array in a (N,3)
    phase = np.reshape(phase,(N,3))
    t_resample = [] 
    
    for i in range(0,N):
        t1,t2,t3 = time_taco[i]
        p1,p2,p3 = phase[i]
        t_matrix = [[1,t1,t1**2],[1,t2,t2**2],[1,t3,t3**2]]
        p_matrix = [0,p2-p1,p3-p1]
        a0,a1,a2 = np.linalg.inv(t_matrix) @ p_matrix
        t_func = lambda theta: (1/(2*a2))*(np.sqrt(np.abs(4*a2*(theta-a0)+a1**2))-a1)
        t_resample.append(t_func) #creating a list of contraints equations according to the relationship between t(theta)
        
    
    phase = np.reshape(phase,M)
    # Resample the signal x(t) into x(theta)
    interp_signal = interp1d(time,signal,kind='linear',fill_value='extrapolate')
    x_resampled = []
    for theta_value in phase:
        t_value = np.array([t_func(theta_value) for t_func in t_resample])[0]
        x_resampled.append(interp_signal(t_value))

    return np.array(x_resampled)


# anti aliasing filter
def anti_aliasing_filter(x,fs,cutoff_freq,filter_order=2):
    """Perform a anti_aliasing_filter

    Args:
        x (array_like): signal to be filtered x(theta)
        fs (int): sampling frequency
        cutoff_freq (int): set your desired cutoff frequency
        filter_order (int): desired filter order: Default = 2

    Returns:
        filtered_signal (ndarray): the output of the digital filter

    """
    
    nyquist = fs/2
    cutoff = cutoff_freq/nyquist
    b, a = signal.butter(filter_order,cutoff, btype = "lowpass")
    filtered_signal = signal.lfilter(b,a,x)
    return filtered_signal
 
    
# FSA => frequency domain averaging

def ftda(x, k,w0):
    """
    Flexible time domain averaging (FTDA) method for filtering and time-differencing approximation of a periodic signal.

    Parameters
    ----------
    x : 1-D numpy array
        Original data to be filtered.
    k: int
        Number of harmonics


    Returns
    -------
    tda : 1-D numpy array
        Discrete FTDA of the filtered signal.
    """
    # number of samples
    N = len(x)

    # frequency vector
    f = np.linspace(w0, 0.5 * N / N, k)
    N_DFT = 2**np.ceil(np.log2(N)).astype(int)
    
    # CZT coefficients
    c= czt(x)

    # set the DC component to zero
    c[0] = 0
    

    # threshold for filtering
    threshold = np.sqrt(2 * np.log(N)) * np.std(x - np.mean(x))
    # threshold for filtering
    c[np.abs(c) < threshold] = 0

    # time vector for TDA
    t = np.arange(0, N / N_DFT, 1 / N_DFT)

    ftda = np.real(np.sum(c * np.exp(2j * np.pi * (f * t[:, None] * np.arange(N_DFT))[None, :]), axis=0))

    return ftda


# Example usage:
if __name__ == '__main__':
    
    """"""""" DELFT
        #initialize Case_1
    path = r"D:\IC EMBRAER\microphones\DELFT_4000_EXPDELFT_J0_0"
    Case_1 = AeroPropCase(path)

        #import variables from microphone
    x = Case_1.microphone.signals[0][:]
    fs = Case_1.microphone.fs #Aquisition frequency signal
    blades = 2 #number of blades
    rpm_estimate = 4000
    BPF = rpm_estimate*blades
    
        #Instant frequency
    band = 0.05 # estimated after fft visual inspection
    f_max = central_freq(x, fs, rpm_estimate, band)

        #MNI
    phase = get_IP_MNI(f_max)
    
        #angular resampling
    x_resampled = angular_resampling(f_max,phase,x)
    
        #FSA
    cutoff_freq = BPF/(8*60)
    df = 30
    f0 = BPF
    filtered_signal = anti_aliasing_filter(x_resampled,fs,cutoff_freq,filter_order=2)
    
  
    
    """""""""

    """""""""SIMULATION
    # Parameters
    A = 0.005
    B = 0.007
    C = 0.006
    phi1 = np.pi / 6
    phi2 = np.pi / 3
    phi3 = np.pi / 2
    f0 = 300
    f1 = 3
    w = 2 * np.pi
    
    fs = 20000 #[Hz]
    t = np.arange(0,5,1/fs)

        # Rotational frequency
    f_t = (300 + 1000 * np.sin(2 * np.pi * 0.1 * t)) / 60

        # Impulse signal
    s = np.exp(-300*t)*np.sin(2*np.pi*2000*t)
    

        # Harmonics
    b1 = A * np.cos(2 * np.pi * f_t * 1 + phi1)
    b2 = B * np.cos(2 * np.pi * f_t * 2 + phi2)
    b3 = C * np.cos(2 * np.pi * f_t * 3 + phi3)

    
        # Simulated signal
    x = b1 + b2 + b3  

    noise_power = 0.1 * np.mean(x**2)  # power of the simulated signal
    noise_power = noise_power * 10**(3/10)  # 3 dB lower power for the desired SNR
    np.random.seed(0)
    noise = np.random.normal(0, np.sqrt(noise_power), len(t))  # white Gaussian noise with the desired power
    x = x + noise

    #simulation
    
        #Instant frequency
    band = 0.05 # estimated after fft visual inspection
    rpm_estimate = 300*60
    f_max = central_freq(x, fs, rpm_estimate, band)
    
        #MNI
    phase = get_IP_MNI(f_max)
    


        #angular resampling
    x_resampled = angular_resampling(f_max,phase,x)
    
    

        #FSA
    

    filt_signal = anti_aliasing_filter(x_resampled,fs,300)
    """""""""
   
    
    

    
        
    
  
    








    
  


