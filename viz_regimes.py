import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_regime_plots(file_path="train_FD002.txt", units=[1, 2], output_dir="plots_regimes"):
    # 1. Load Data
    columns = ['unit', 'time', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(file_path, sep=r'\s+', names=columns)
    
    # 2. Identify Regimes
    # The 6 regimes in FD002 are combinations of (os1, os2, os3)
    # Rounding helps handle small floating point variances
    df['regime_key'] = df[['os1', 'os2', 'os3']].round(1).apply(tuple, axis=1)
    unique_regimes = df['regime_key'].unique()
    regime_map = {regime: i+1 for i, regime in enumerate(unique_regimes)}
    df['regime_id'] = df['regime_key'].map(regime_map)
    
    # 3. Setup Output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Use 14 informative sensors to keep the plot manageable
    sensor_cols = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'] # 6 distinct colors
    
    for unit_id in units:
        unit_df = df[df['unit'] == unit_id].copy()
        
        # Create a large plot with 5 rows and 3 columns (14+ sensors)
        fig, axes = plt.subplots(5, 3, figsize=(20, 25))
        fig.suptitle(f"Color-Coded Regimes for FD002 Unit {unit_id}", fontsize=25)
        
        for i, sensor in enumerate(sensor_cols):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Plot each regime as a scatter plot with its own color
            for r_id in range(1, 7):
                r_mask = unit_df['regime_id'] == r_id
                if r_mask.any():
                    ax.scatter(unit_df[r_mask]['time'], unit_df[r_mask][sensor], 
                               c=colors[r_id-1], label=f"Regime {r_id}", s=10, alpha=0.7)
            
            ax.set_title(f"Sensor {sensor}", fontsize=15)
            ax.set_xlabel("Cycles")
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Only add legend to the first plot to save space
            if i == 0:
                ax.legend(loc='upper left', fontsize=10)
                
        # Hide the empty 15th subplot
        axes[4, 2].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save Plot
        file_name = f"unit_{unit_id}_regime_colors.png"
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path)
        plt.close()
        print(f"Generated regime plot for Unit {unit_id} at {save_path}")

if __name__ == "__main__":
    if os.path.exists("train_FD002.txt"):
        generate_regime_plots()
    else:
        print("Data file 'train_FD002.txt' not found.")
