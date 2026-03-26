import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_sensor_plots(file_path="train_FD001.txt", units=[1, 2], output_dir="plots_FD001"):
    # 1. Load Data
    columns = ['unit', 'time', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(file_path, sep=r'\s+', names=columns)
    
    # 2. Setup Output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sensor_cols = [f's{i}' for i in range(1, 22)]
    
    for unit_id in units:
        unit_df = df[df['unit'] == unit_id]
        
        # Create a large plot with 7 rows and 3 columns (21 sensors)
        fig, axes = plt.subplots(7, 3, figsize=(20, 30))
        fig.suptitle(f"Sensor Trends for Engine Unit {unit_id}", fontsize=25)
        
        for i, sensor in enumerate(sensor_cols):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            ax.plot(unit_df['time'], unit_df[sensor], color='blue')
            ax.set_title(f"Sensor {sensor}", fontsize=15)
            ax.set_xlabel("Cycles")
            ax.grid(True, linestyle='--', alpha=0.7)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save Plot
        file_name = f"unit_{unit_id}_sensors.png"
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path)
        plt.close()
        print(f"Generated plot for Unit {unit_id} at {save_path}")

if __name__ == "__main__":
    import sys
    # Check if we have the data
    if os.path.exists("train_FD001.txt"):
        generate_sensor_plots()
    else:
        print("Data file 'train_FD001.txt' not found.")
