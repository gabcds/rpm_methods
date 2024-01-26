import csv
import numpy as np
import h5py
import pandas as pd
import os

#csv
def read_csv_with_csv_module(file_path):
    arr = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)  # Skip the header
        for row in reader:
            arr.append(row)
    return np.array(arr)

def read_csv_with_numpy(file_path):
    arr = np.loadtxt(file_path, delimiter=',', skiprows=1)
    return arr


if __name__ == '__main__':
    read_csv_with_csv_module(r"D:\IC EMBRAER\microphones\rpm.csv")
    
#h5

def load_data_h5(path):
    # Open the .h5 file
    with h5py.File(path, 'r') as f:
        # Create an empty dictionary to hold arrays (optional)
        data_dict = {}

        # Function to recursively read groups and datasets
        def recursive_read(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Read the data
                value = obj[...]

                # Convert to DataFrame
                df = pd.DataFrame({name: value.flatten()})

                # Cut the .h5 file name from path
                # (e.g., 'data/file.h5' becomes 'file')
                file_name = path.split('/')[-1].split('.')[0]

                # Create directory if it doesn't exist
                if not os.path.exists(f'{file_name}'):
                    os.makedirs(f'{file_name}')

                # Save DataFrame to CSV file
                df.to_csv(f'{file_name}/{name.replace("/", "_")}_dataset.csv', index=False)

        # Iterate through each key in the .h5 file
        f.visititems(recursive_read)
