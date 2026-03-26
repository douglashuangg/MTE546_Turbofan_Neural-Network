import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class CMAPSSDataset(Dataset):
    def __init__(self, file_path, sequence_length=30, missing_prob=0.1, noise_level=0.1):
        """
        Parses the NASA Turbofan dataset and generates sequences per engine.
        Applies synthetic faults (dropout and noise).
        """
        self.sequence_length = sequence_length
        self.missing_prob = missing_prob
        self.noise_level = noise_level
        
        # Load data
        # C-MAPSS columns: unit, time, 3 op settings, 21 sensors
        columns = ['unit', 'time', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
        df = pd.read_csv(file_path, sep=r'\s+', names=columns)
        
        # Standard 14 informative sensors for C-MAPSS
        self.sensor_cols = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
        
        # Calculate RUL for each engine unit
        # Piecewise Linear RUL: Clip at 125 (industry standard)
        rul = pd.DataFrame(df.groupby('unit')['time'].max()).reset_index()
        rul.columns = ['unit', 'max']
        df = df.merge(rul, on=['unit'], how='left')
        df['RUL'] = df['max'] - df['time']
        df['RUL'] = df['RUL'].clip(upper=125) # RUL Clipping
        df.drop('max', axis=1, inplace=True)
        
        # Operating Condition-Aware Normalization
        # Round the operating settings to identify the 6 distinct conditions
        df['cond'] = df[['os1', 'os2', 'os3']].round(1).apply(tuple, axis=1)
        
        for col in self.sensor_cols:
            # Scale within each condition group
            def cond_min_max(group):
                g_min = group.min()
                g_max = group.max()
                if g_max - g_min > 1e-9:
                    return (group - g_min) / (g_max - g_min)
                else:
                    return group - group # Return zeros if sensor is constant in this condition
            
            df[col] = df.groupby('cond')[col].transform(cond_min_max)
        
        df.drop('cond', axis=1, inplace=True)
            
        # Generate Sequences
        self.sequences = []
        self.labels = []
        
        for unit_id in df['unit'].unique():
            unit_data = df[df['unit'] == unit_id]
            # Time series sliding window
            for i in range(len(unit_data) - sequence_length + 1):
                window = unit_data.iloc[i:i + sequence_length]
                sensors = window[self.sensor_cols].values # shape (sequence_length, 3)
                label = window['RUL'].iloc[-1]
                
                self.sequences.append(sensors)
                self.labels.append(label)
                
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)

    def apply_synthetic_faults(self, sensor_data):
        """
        Takes sensor_data of shape (sequence_length, 3)
        Applies dropout and non-linear noise randomly.
        """
        faulty_data = sensor_data.copy()
        
        for i in range(faulty_data.shape[1]): # Iterate over all sensors
            # Apply Dropout
            if np.random.rand() < self.missing_prob:
                # Mask out completely
                faulty_data[:, i] = 0.0
            else:
                # Apply non-linear noise (e.g. exponential or heteroscedastic)
                noise = np.random.normal(0, self.noise_level, size=self.sequence_length)
                # Introduce non-linearity
                faulty_data[:, i] += noise * np.abs(faulty_data[:, i])
                
        return faulty_data

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        clean_sensors = self.sequences[idx] # (seq_len, num_sensors)
        label = self.labels[idx]
        
        # Apply faults
        faulty_sensors = self.apply_synthetic_faults(clean_sensors)
        
        # PyTorch expects input as (num_sensors, sequence_length) normally,
        # but our model lists inputs. 
        # Return as a tensor of shape (num_sensors, seq_len)
        data = torch.tensor(faulty_sensors.T, dtype=torch.float32)
        target = torch.tensor([label], dtype=torch.float32)
        
        return data, target

if __name__ == "__main__":
    # Test dataset
    # You must have train_FD002.txt in the same directory
    import os
    if os.path.exists("train_FD002.txt"):
        print("Loading dataset...")
        dataset = CMAPSSDataset("train_FD002.txt", sequence_length=30)
        print(f"Total sequences generated: {len(dataset)}")
        x, y = dataset[0]
        print(f"X shape: {x.shape}, Y shape: {y.shape}")
