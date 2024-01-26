import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate
from scipy.interpolate import interp1d
from LVA_techniques.rpm_extract import *



#fs = 1000 
#rpm_estimate = 4000 #[rpm]
#samples = len(x)
#time = np.linspace(0, samples/fs, samples, endpoint=False)


def fft_visual_inspection(fs, x,  BPF_rotation,font_size = 20):
    """plot the spectrum from the signal

    Args:
        fs (float): sample frequency
        x (array_like): signal
        BPF_rotation (float): estimated BPF rotation
    """
    BPF= BPF_rotation/60 
    nperseg = fs/(BPF /8)
     # Ensure that there's at least one segment
    nperseg = max(nperseg, 1)

    # Compute the number of overlapping samples
    noverlap = int(0.9*nperseg) // 4

    sample_freq, spl= signal.welch(x, fs, nperseg=nperseg, noverlap=noverlap, scaling='spectrum', average='mean')
    plt.plot(sample_freq, spl)
    plt.xscale('log')
    plt.xlim(10,1E4)
    plt.ylim(spl.min()-5,spl.max()+5)
    plt.xlabel("Hz", fontsize=font_size)
    plt.ylabel("SPL [delta f=1/8 BPF Hz]", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2, fontsize=font_size)

band = 0.05 # estimated after visual inspection

# (1) Finding the IF (intantaneous frequency) of the raw signal
def central_freq(x, fs, rpm_estimate, band):
    return demodulation_method(x, fs, rpm_estimate, band)

def get_IP_CNI(f_max,fs):
    """Perform the numerical integration to evaluate the preliminary inst. phase 

    Args:
        f_max (array_like): array of the inst frequency
        fs (float): sampling frequency

    Returns: 
        inst_phase (array_like): Inst. phase matrix
    """
    time = np.linspace(0, samples/fs, samples, endpoint=False)

    interp_f_max = interp1d(time,f_max,kind="cubic")
    tn = 1/fs
    #theta_11 = abs(tn*(f_max[1]-f_max[tn])/4) theta_11 is first aproximated by CNI
    #loop starts from the second term -> the delta_phase starts at zero
    inst_phase = []
    h = tn #trapezoid lenght
    for i in range(tn,len(f_max),tn):
        area = (f_max[i] + f_max[i-tn])*tn/2
        inst_phase.append(area) 
    return inst_phase
    

def get_IP_MNI(f_max,fs):


    """Obtain the corrected instantaneus phase (theta_jj) by MNI
    Args:
        f_max (array_like): Inst. frequency matrix
        fs (float): sampling frequency [Hz]

    Returns: 
        inst_phase_cor (array_like): Corrected inst. phase matrix
        
    """ 
    time = np.linspace(0, samples/fs, samples, endpoint=False)
    interp_f_max = interp1d(time,f_max,kind="cubic")  
    tn = 1/fs 
    theta_11 = abs(tn*(f_max[1]-f_max[tn])/4) #theta_11 is first aproximated by CNI
    inst_phase_cor = [theta_11] #corrected inst. phase
    a = tn
    b = tn + tn
    for _ in range(tn,len(f_max),tn): #start from the second time instant
        f_max_romb_integral = integrate.romberg(interp_f_max,a,b, show = "False") #MNI for theta_jj estimation
        inst_phase_cor.append(f_max_romb_integral) #theta_jj
        a += tn
        b += tn
        inst_phase_cor = np.array(inst_phase_cor)
        
    return inst_phase_cor

def get_time_duration(inst_phase_cor,inst_phase,fs):
    """Get the time duration between phase variation

    Args:
        inst_phase (array_like): Inst. phase matrix         
        fs (float): Sampling frequency [Hz] 

    Returns: 
        n (int): elapsed points
        delta_theta_interp: theta(t)
    """
    time = np.linspace(0, samples/fs, samples, endpoint=False)
    tn = 1/fs
    delta_theta = inst_phase - inst_phase_cor #must have the same lenght 
    delta_t = interp1d(delta_theta,time,kind = "linear")  #time variation between consecutive phase variation
    #interpolate with the time instants being the y and the delta theta being the x
    n = delta_t*fs
    return n,delta_theta

# anti aliasing filter
def anti_aliasing_filter(x,fs,cutoff_freq,filter_order):
    """Perform a anti_aliasing_filter

    Args:
        x (array_like): signal to be filtered
        fs (int): sampling frequency
        cutoff_freq (int): set your desired cutoff frequency
        filter_order (int): desired filter order

    Returns:
        filtered_signal (ndarray): the output of the digital filter

    """
    #t = np.linspace(0, 1, fs, endpoint=False)
    nyquist = fs/2
    cutoff = cutoff_freq/nyquist
    b, a = signal.butter(filter_order,cutoff, btype = "lowpass")
    filtered_signal = signal.lfilter(b,a,x)
    return filtered_signal

#(3) Angular resampling


def angular_resampling(time,delta_theta,filtered_signal):

    """Perform a angular resampling based on 3 consecutive points according to Taylor polynomial

    Args:
        time (array_like): time samples
        delta_theta (array_like): theta(t) before resampling
        filtered_signal (array_like): signal after anti aliasing filtering
        
    Returns:
        signal resampled function
    """
    consec_times = [time[1],time[2],time[3]]
    consec_phases = [delta_theta[1],delta_theta[2],delta_theta[3]] #consec_phases[i] must match consec_time[i]
    phi_matrix = [0,delta_theta[2]-delta_theta[1],delta_theta[3]-delta_theta[1]]
    phi = [[x] for x in phi_matrix]
    time_matrix = [[1,1,1],[time[1],time[2],time[3]],[time[1]**2,time[2]**2,time[3]**2]]
    a = np.linalg.inv(time_matrix) @ phi #find the coeficients for angular resampling
    #time and delta_theta being np.arrays!
    #time_resampled = (1/2*a[2]) * (np.sqrt(4*a[2]*(delta_theta-a[0])+a[2])) - a[1] 
    theta_resampled = ((2*a[2]*time + a[1])**2 - a[1]**2 +4*a[2]*a[0])/(4*a[2]) 
    return interp1d(theta_resampled,filtered_signal,kind = 'linear')


# (4) FSA for denoising