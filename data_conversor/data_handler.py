import csv
import os
import numpy as np
import h5py

class AeroPropCase:
    class Microphones:
        def __init__(self):
            self.number = None
            self.positions = []
            self.signals = []  # List of arrays
            self.spectra = []  # List of arrays
            self.frequencies = []  # List of arrays
            self.resampled_signals = []
            self.fs = None  # Acquisition frequency

    class Shaft:
        def __init__(self):
            self.rpm_inst = []
            self.encoder = []
            self.time_taco = []
            self.fx = []
            self.fy = []
            self.fz = []
            self.tx = []
            self.ty = []
            self.tz = []

            self.fs = None  # Acquisition frequency

    class Metadata:
        def __init__(self):
            self.propeller_type = None
            self.num_blades = None
            self.target_rpm = None
            self.noise_calculation_method = None
            self.software = None
            self.advance_ratio = None

    def __init__(self, directory):
        self.directory = directory
        self.microphone = self.Microphones()
        self.shaft = self.Shaft()
        self.metadata = self.Metadata()
        self.process_files()

    def read_csv(self, filepath):
        data = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        return data

    def process_files(self):
        for filename in os.listdir(self.directory):
            filepath = os.path.join(self.directory, filename)
            if filename.endswith(".h5"):
                if 'microphones' in filename:
                    self.process_microphones_data(filepath)

                if 'shaft' in filename:
                    self.process_shaft_data(filepath)
                    

    
    def process_microphones_data(self, filepath):
        with h5py.File(filepath, 'r') as file:
            # Check for the existence of attributes and assign if they exist
            if 'mics_number' in file.attrs:
                self.microphone.number = file.attrs['mics_number']
            if 'mics_fs' in file.attrs:
                self.microphone.fs = file.attrs['mics_fs']

            # Initialize lists
            self.microphone.positions = []
            self.microphone.signals = []
            self.microphone.spectra = []
            self.microphone.frequencies = []

            # Iterate through each microphone group, check if it exists before accessing
            for i in range(1, self.microphone.number + 1):
                mic_key = f'mic{i}'
                if mic_key in file:
                    mic_group = file[mic_key]

                    if 'position' in mic_group:
                        position = list(map(float, mic_group['position']))
                        self.microphone.positions.append(position)

                    if 'signal' in mic_group:
                        signal = np.array(mic_group['signal'])
                        self.microphone.signals.append(signal)

                    if 'spectrum' in mic_group:
                        spectra = np.array(mic_group['spectrum'])
                        self.microphone.spectra.append(spectra)

                    if 'frequencies' in mic_group:
                        frequency = np.array(mic_group['frequencies'])
                        self.microphone.frequencies.append(frequency)
                # Check if 'signal_resampled' exists in the mic_group
                if 'signal_resampled' in mic_group:
                    signal_resampled = np.array(mic_group['signal_resampled'])
                    self.microphone.resampled_signals.append(signal_resampled)
                else:

                    self.microphone.resampled_signals.append(None)
    # Add other processing functions similar to `process_microphones_data`
    def process_shaft_data(self, filepath):
        with h5py.File(filepath, 'r') as file:
            # Global attributes
            self.shaft.fs = file.attrs['fs_taco']

            # Iterate through each force and torques
            #if 'rpm_inst' in file:
                #self.shaft.rpm_inst = np.array(file['rpm_inst'])
            #if 'encoder' in file:
            self.shaft.encoder = np.array(file['encoder'])
            self.shaft.time_taco = np.array(file['Time taco'])
            for x in ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]:
                if x in file:
                    setattr(self.shaft, x.lower(), np.array(file[x]))
            

                    
            

# Example usage:
if __name__ == '__main__':
    cases_path = '/run/media/mateus/Disk/Data/UFSC/Doutorado/Results/Dados/raw_data/Noise_signal/'

    Test = AeroPropCase(cases_path + 'DELFT_mesh-2_'+str(5000)+'_VLES_optydb_prm_J0_0')
    hz = np.array(Test.microphone.frequencies[0][:])
    spl = np.array(Test.microphone.spectra[0][:])
    print(spl)