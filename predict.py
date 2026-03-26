import torch
import pandas as pd
import numpy as np
from fusion_model import ReliabilityWeightedFusion

def predict_on_test_set(test_data_path="test_FD002.txt", rul_data_path="RUL_FD002.txt", model_path="fusion_model.pth", sequence_length=30):
    # 1. Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_sensors = 14
    model = ReliabilityWeightedFusion(num_sensors=num_sensors, sequence_length=sequence_length).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model from {model_path}.")
    except Exception as e:
        print(f"Could not load model. Did you run train.py first? Error: {e}")
        return

    model.eval()
    
    # 2. Load and Prepare Test Data
    columns = ['unit', 'time', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    df_test = pd.read_csv(test_data_path, sep=r'\s+', names=columns)
    
    # Ground truth RUL values for each unit at the last time step
    true_ruls = pd.read_csv(rul_data_path, sep=r'\s+', header=None).values.flatten()
    
    sensor_cols = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    
    # Min-Max Scaling (using local test stats for simplicity, ideally use train stats)
    for col in sensor_cols:
        df_test[col] = (df_test[col] - df_test[col].min()) / (df_test[col].max() - df_test[col].min() + 1e-9)
        
    predictions = []
    
    # Ground truth RUL values for each unit at the last time step
    # Industry standard: eval metrics often use clipped ground truth as well
    true_ruls = np.clip(true_ruls, 0, 125) 
    
    print(f"\nPredicting RUL for {len(units)} engines in the test set...\n")
    print("-" * 50)
    print(f"{'Engine Unit':<15} | {'Predicted RUL':<15} | {'True RUL':<10}")
    print("-" * 50)
    
    mses = []
    
    for i, unit_id in enumerate(units):
        unit_data = df_test[df_test['unit'] == unit_id]
        
        # We need at least 'sequence_length' cycles to make a prediction
        if len(unit_data) >= sequence_length:
            window = unit_data.iloc[-sequence_length:]
            sensors = window[sensor_cols].values # shape (sequence_length, 3)
            
            # Format shape to (num_sensors, sequence_length) -> PyTorch input shape: (1, num_sensors, seq_len)
            data = torch.tensor([sensors.T], dtype=torch.float32).to(device)
            
            sensor_inputs = [data[:, j, :] for j in range(len(sensor_cols))]
            
            with torch.no_grad():
                predicted_rul, weights = model(sensor_inputs)
                pred_val = predicted_rul.item()
                
            true_val = true_ruls[i]
            
            predictions.append(pred_val)
            mses.append((pred_val - true_val)**2)
            
            # Print the first 10 for display
            if i < 10:
                print(f"Unit {unit_id:<10} | {pred_val:<15.2f} | {true_val:<10.2f}")
    
    if len(units) > 10:
        print(f"... and {len(units)-10} more engines.")
    
    # 3. Calculate metrics
    rmse = np.sqrt(np.mean(mses))
    print(f"\nFinal Test RMSE: {rmse:.2f}")

if __name__ == "__main__":
    predict_on_test_set()
