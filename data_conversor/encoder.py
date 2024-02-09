import numpy as np
def moving_average(data, window_size):
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

def get_rpm_from_enconder(timedata, enconder_data, window_size=100):
    """
    Get the rpm from the enconder data
    """

    # Guessing the PPR (Pulses Per Revolution) from the enconder
    PPR = 200
    rotations = [count / PPR for count in enconder_data]

    rpm = np.zeros(len(enconder_data))
    
    for x in range(len(enconder_data)):
        rpm[x] = 60 * (rotations[x] - rotations[x-1]) / (timedata[x] - timedata[x-1])
    # Using a floating average for avoiding outlayers
    return moving_average(rpm, window_size)
