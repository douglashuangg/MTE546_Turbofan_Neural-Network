import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_multi_sample_red_plots(file_path="train_FD002.txt", units=[3, 10, 20, 30, 50], target_regime=4, color='#d62728', color_name="red", output_dir="plots_red_samples"):
    # 1. Load Data
    columns = ['unit', 'time', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(file_path, sep=r'\s+', names=columns)
    
    # 2. Identify Regimes exactly like the previous script
    df['regime_key'] = df[['os1', 'os2', 'os3']].round(1).apply(tuple, axis=1)
    unique_regimes = df['regime_key'].unique()
    regime_map = {regime: i+1 for i, regime in enumerate(unique_regimes)}
    df['regime_id'] = df['regime_key'].map(regime_map)
    
    # 3. Filter for TARGET regime
    df_single = df[df['regime_id'] == target_regime].copy()
    
    # 4. Setup Output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sensor_cols = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    
    for unit_id in units:
        unit_df = df_single[df_single['unit'] == unit_id]
        if unit_df.empty:
            print(f"No data for Unit {unit_id} in {color_name.capitalize()} regime. Skipping.")
            continue
            
        # Create a large plot
        fig, axes = plt.subplots(5, 3, figsize=(20, 25))
        fig.suptitle(f"Single Regime (Red) for FD002 Unit {unit_id}", fontsize=25)
        
        for i, sensor in enumerate(sensor_cols):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Scatter plot + Line to show the smooth trend
            ax.scatter(unit_df['time'], unit_df[sensor], c=color, s=20, alpha=0.6)
            ax.plot(unit_df['time'], unit_df[sensor], c=color, alpha=0.3)
            
            ax.set_title(f"Sensor {sensor}", fontsize=15)
            ax.set_xlabel("Cycles")
            ax.grid(True, linestyle='--', alpha=0.5)
            
        axes[4, 2].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save Plot
        file_name = f"unit_{unit_id}_red_regime.png"
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path)
        plt.close()
        print(f"Generated Red regime plot for Unit {unit_id} at {save_path}")

if __name__ == "__main__":
    if os.path.exists("train_FD002.txt"):
        generate_multi_sample_red_plots()
    else:
        print("Data file 'train_FD002.txt' not found.")
