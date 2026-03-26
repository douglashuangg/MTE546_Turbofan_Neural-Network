from viz_fd001 import generate_sensor_plots
import os

if __name__ == "__main__":
    if os.path.exists("train_FD002.txt"):
        generate_sensor_plots(file_path="train_FD002.txt", units=[1, 2], output_dir="plots_fd002")
    else:
        print("Data file 'train_FD002.txt' not found.")
