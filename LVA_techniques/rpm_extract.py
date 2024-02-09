# Autor(a): RACQUEL KNUST

import numpy as np
from scipy import signal, interpolate
import json
import matplotlib.pyplot as plt

fs = 102400
def rpm_detection(waveform, fs, rpm_estimate, band=30, method="hybrid"):
    """Detection of rotation speed

    Args:
        waveform (list): acceleration signal
        fs (int): sampling frequency
        rpm_estimate (float): estimated rotation speed in rpm
        band (float, optional):
            frequency band in percentage of     the estimated. Defaults to 30.
        method (str, optional):
            method of speed determination:
            "hybrid", "demodulation" or "spectrogram". Defaults to "hybrid".

    Returns:
        int: rotation speed determination in rpm
    """
    waveform = np.array(waveform)-np.mean(waveform)
    if method == "hybrid":
        return hybrid_method(waveform, fs, rpm_estimate, band)
    elif method == "demodulation":
        return demodulation_method(waveform, fs, rpm_estimate, band)
    elif method == "spectrogram":
        return spectrogram_method(waveform, fs, rpm_estimate, band)


def demodulation_method(waveform, fs, rpm_estimate, band):
    """Demodulation technique to determine the rotation speed

    Args:
        waveform (array_like): acceleration waveform
        fs (int): sampling frequency
        rpm_estimate (float): estimated rotation speed in rpm
        band (float): frequency band in percentage of the estimated
            rotation speed.

    Returns:
        list: frequencyy per rotation at each time instant
    """
    lower_band_limit = rpm_estimate/60*(1-band/100)
    higher_band_limit = rpm_estimate/60*(1+band/100)

    signal_filtered = bandpass_filter(
        waveform,
        fs,
        lower_band_limit,
        higher_band_limit
    )
    analitical_signal = signal.hilbert(signal_filtered)
    analitical_signal_phase = np.unwrap(np.angle(analitical_signal))

    samples = len(waveform)
    time = np.linspace(0, samples/fs, samples, endpoint=False)
    techometer_time_instants = tachometer_analitical_phase(
        analitical_signal_phase,
        time
    )

    freq_per_rotation = frequency_per_rotation(techometer_time_instants)

    return freq_per_rotation


def spectrogram_method(waveform, fs, rpm_estimate, band):
    """Spectrogram technique to determine the rotation speed

    Args:
        waveform (array_like): acceleration waveform
        fs (int): sampling frequency
        rpm_estimate (float): estimated rotation speed in rpm
        band (float): frequency band in percentage of the estimated
            rotation speed.

    Returns:
        int: rotation speed detected in rpm
    """
    freq, _, stfts = get_spectrogram(waveform, fs)
    freq_max = freq_maximum_of_each_window(
        stfts=stfts,
        delta_freq=freq[1],
        rpm_estimate=rpm_estimate,
        band=band
    )
    return freq_max

def hybrid_method(waveform, fs, rpm_estimate, band):
    """Hybrid (sepectrogram+demodulation) technique to determine the
    rotation speed

    Args:
        waveform (array_like): acceleration waveform
        fs (int): sampling frequency
        rpm_estimate (float): estimated rotation speed in rpm
        band (float): frequency band in percentage of the estimated
            rotation speed.

    Returns:
        int: rotation speed detected in rpm
    """
    samples = len(waveform)
    time = np.linspace(0, samples/fs, samples, endpoint=False)
    # Spectrogram method
    freq, time_segments, stfts = get_spectrogram(waveform, fs)
    freq_max = freq_maximum_of_each_window(
        stfts=stfts,
        delta_freq=freq[1],
        rpm_estimate=rpm_estimate,
        band=band
    )
    # Moving to angle domain
    angle, time_of_angle = get_angle_for_each_intant(
        time,
        time_segments,
        freq_max
    )
    rotate_time_complete = get_time_for_each_rotation(
        angle=angle,
        time_of_angle=time_of_angle
    )
    (waveform_angle_dom,
     time_at_angle_increments,
     angle_resolution) = move_to_angle_domain(
        waveform,
        time,
        rotate_time_complete
    )

    # Demodulation method
    signal_filtered = bandpass_filter(
        waveform=waveform_angle_dom,
        fs=(2*np.pi/angle_resolution),
        lower_band_limit=0.98,
        higher_band_limit=1.02
    )
    analitical_signal = signal.hilbert(signal_filtered)
    analitical_signal_phase = np.unwrap(np.angle(analitical_signal))
    techometer_time_instants = tachometer_analitical_phase(
        analitical_signal_phase=analitical_signal_phase,
        time=time_at_angle_increments

    )
    freq_per_rotation = frequency_per_rotation(techometer_time_instants)

    return  freq_per_rotation


def bandpass_filter(waveform, fs, lower_band_limit, higher_band_limit):
    """Function to apply a bandpass filter

    Args:
        waveform (array_like): acceleration signal
        rpm_estimate (float): estimated rotation speed in rpm
        band (float): frequency band in percentage of the estimated.

    Returns:
        ndarray: filtered waveform
    """
    sos_coef = signal.butter(
        N=3,
        Wn=[lower_band_limit, higher_band_limit],
        btype='bandpass',
        fs=fs,
        output='sos'
    )
    return signal.sosfilt(sos_coef, waveform)


def tachometer_analitical_phase(analitical_signal_phase, time):
    """Determine all time instants at whcih the phase completes one full
        rotation (2pi)

    Args:
        analitical_signal_phase (array_like): phase of analytical signal
        fs (int): sampling frequency

    Returns:
        ndarray: time instant of rotation completion
    """
    phase_time_interpolate = interpolate.interp1d(
        analitical_signal_phase,
        time,
        'linear'
    )
    increment_array_2pi = np.arange(
        2*np.pi,
        analitical_signal_phase[-1],
        2*np.pi
    )
    return phase_time_interpolate(increment_array_2pi)


def frequency_per_rotation(techometer_time_instants):
    """Determine the frequency in hertz for each full rotarion of the phase

    Args:
        techometer_time_instants (array_like):
            time instants of rotation completion

    Returns:
        ndarray: frequency of each rotation
    """
    return [
        1 / (tacho_next_instant - tacho_intant)
        for tacho_intant, tacho_next_instant in zip(
            techometer_time_instants, techometer_time_instants[1:]
        )
    ]


def get_spectrogram(waveform, fs):
    """Obtain spectrogram

    Args:
        waveform (array_like): acceleration signal
        fs (int): sampling frequency

    Returns:
        frequency (ndarray): array of frequencies of STFTs
    """
    window_lenght = int(len(waveform)/1.5)
    overlap = (90/100)*window_lenght
    return signal.spectrogram(
        x=waveform,
        fs=fs,
        window='hann',
        nperseg=window_lenght,
        noverlap=overlap,
        scaling='spectrum'
    )

def freq_maximum_of_each_window(stfts, delta_freq,
                                rpm_estimate, band):
    """Build an array of the frequencies of the maximum STFT at each segment

    Args:
        stfts (array_like): short time fourier transforms matix
        delta_freq (float): frequency resolution
        rpm_estimate (float): estimated rotation speed in rpm
        band (float): frequency band in percentage of the estimated
            rotation speed.

    Returns:
        ndarray: array of frequencies of maximum amplitude for each
        stft segment
    """
    f_max = [rpm_estimate/60]
    for segment, stft in enumerate(stfts.T[1:]):
        low_band_index = round((f_max[segment-1]*(1-band/100))/delta_freq)
        high_band_index = round((f_max[segment-1]*(1+band/100))/delta_freq)
        index_of_max = np.argmax(stft[low_band_index:high_band_index+1])
        f_max.append(delta_freq*(index_of_max+low_band_index))
    return f_max

def get_angle_for_each_intant(time, time_segments, freq_max):
    """Get angle of rotation for each time instant

    Args:
        time (array_like): time array
        time_segments (float): time segments of the STFT
        freq_max (float): maximum frequency of each window of STFT

    Returns:
        angle (ndarray): angle of rotation for each time instant
        time_of_angle (ndarray): time referent to each angle
    """
    fs = 1/time[1]
    time_of_angle = time[
        int(time_segments[0]/time[1]):int(time_segments[-1]/time[1])+1
    ]
    fmax_time_interpolate = interpolate.interp1d(
        time_segments,
        freq_max,
        'linear',fill_value="extrapolate"
    )

    fmaxes_at_instant = fmax_time_interpolate(time_of_angle)
    angle = [0]
    angle.extend(
        angle[-1] + (2 * np.pi * fmax_at_instant) / fs
        for fmax_at_instant in fmaxes_at_instant[:-1]
    )

    return angle, time_of_angle

def get_time_for_each_rotation(angle, time_of_angle):
    """time of each rotation completion

    Args:
        angle (array_like): angle of rotation for each time instant
        time_of_angle (array_like): time referent to each angle

    Returns:
        ndarray: time array of each rotation completion
    """
    angle_time_interpolate = interpolate.interp1d(
        angle,
        time_of_angle,
        'linear'
    )
    return angle_time_interpolate(
        np.arange(2*np.pi, angle[-1], 2*np.pi)
    )
    # for time_of_complet in instant_of_rotation_complet:
    #     if time_of_complet > measure_duration:
    #         time_of_complet = 0
    # return list(filter((0).__ne__, instant_of_rotation_complet))

def move_to_angle_domain(waveform, time, rotate_time_complete):
    """Move to angle domain by finding the mean frequency to segment the
    angles acordingin to it and than resample the waveform according to
    the time instant at each angle segment.

    Args:
        waveform (array_like): acceleration signal
        time (array_like): time referent to signal
        rotate_time_complete (array_like):
            array of instants of roataion complete

    Returns:
        ndarray: waveform resampled by the angle
        time_at_angle_increments (ndarray):
            time relative to the resampled waveform
        angle_resolution (float): angle segments length
    """
    f_med = np.mean(
        [1/(next-current) for current, next in
         zip(rotate_time_complete, rotate_time_complete[1:])]
    )
    # axis markings - Division of the full circle (2pi) in equal segments
    # based in the mean frequency rotation
    angle_resolution = (2*np.pi)/int(fs/f_med)
    angle_increments = np.arange(
        0,
        2*np.pi*(len(rotate_time_complete)-1),
        angle_resolution
    )
    increments_2pi = np.linspace(0, 2*np.pi*(len(rotate_time_complete)-1),
                                 num=len(rotate_time_complete))
    time_2pincrement_interp = interpolate.interp1d(
        increments_2pi,
        rotate_time_complete,
        'linear'
    )
    time_at_angle_increments = time_2pincrement_interp(angle_increments)
    original_waveform_interp = interpolate.interp1d(time, waveform, 'linear')
    return (
        original_waveform_interp(time_at_angle_increments),
        time_at_angle_increments,
        angle_resolution
    )


if __name__ == "__main__":

    with open('temp/test_data.json', "r") as f:
        test_data = json.load(f)

    acc_signal = np.array(test_data["input"]["signal"])
    fs = test_data["input"]["fs"]
    lowbw = test_data["input"]["lowbw"]
    highbw = test_data["input"]["highbw"]
    samples = len(acc_signal)
    tempo = np.linspace(0, samples/fs, samples, endpoint=False)
    rpm = test_data["input"]["rpm_estimated"]

    demo_test_result = test_data["output"]["demodualtion"]
    spec_test_result = test_data["output"]["spectrogram"]
    hyb_test_result = test_data["output"]["hybrid"]

    demo = rpm_detection(
        acc_signal,
        fs,
        rpm,
        band=30,
        method="demodulation"
    )
    spec = rpm_detection(
        acc_signal,
        fs,
        rpm,
        band=30,
        method="spectrogram"
    )
    hyb = rpm_detection(
        acc_signal,
        fs,
        rpm,
        band=30,
        method="hybrid"
    )

    print(f"demo - {demo_test_result == demo}")
    print(f"spec - {spec_test_result == spec}")
    print(f"hyb - {hyb_test_result == hyb}")


    
