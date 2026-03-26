import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CMAPSSDataset
from fusion_model import ReliabilityWeightedFusion

def nasa_scoring_function(y_pred, y_true):
    """
    Calculates the asymmetric NASA scoring function.
    Late predictions (predicting a longer life than reality) are heavily penalized.
    d = y_pred - y_true
    score = sum( exp(-d/13) - 1 ) for d < 0 (early predictions)
          + sum( exp(d/10) - 1 ) for d >= 0 (late predictions)
    """
    d = y_pred - y_true
    
    # Masks for early and late predictions
    early_mask = d < 0
    late_mask = d >= 0
    
    early_score = torch.sum(torch.exp(-d[early_mask] / 13.0) - 1)
    late_score = torch.sum(torch.exp(d[late_mask] / 10.0) - 1)
    
    # Make sure we don't get nans if a mask is empty
    score = (early_score if early_score.numel() > 0 else 0) + \
            (late_score if late_score.numel() > 0 else 0)
            
    return score

def train_model(data_path="train_FD002.txt", epochs=100, batch_size=64, lr=0.001):
    # Setup Data
    dataset = CMAPSSDataset(data_path, sequence_length=30, missing_prob=0.1, noise_level=0.1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset size: {len(dataset)} | Batches: {len(dataloader)}")

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReliabilityWeightedFusion(num_sensors=14, sequence_length=30).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    mse_criterion = nn.MSELoss()
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_nasa_score = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # data shape: (batch_size, num_sensors, seq_len)
            data, target = data.to(device), target.to(device)
            
            # Split into list of tensors per sensor for the model
            num_sensors = data.shape[1]
            sensor_inputs = [data[:, i, :] for i in range(num_sensors)]
            
            optimizer.zero_grad()
            
            # Forward pass
            rul_pred, weights = model(sensor_inputs)
            
            # Calculate Losses
            rmse_loss = torch.sqrt(mse_criterion(rul_pred, target))
            nasa_score = nasa_scoring_function(rul_pred, target)
            
            # Total loss balances RMSE and the penalizing NASA score
            # A common approach is optimizing RMSE primarily, tracking NASA score.
            # Or optimizing a sum. We will train primarily on RMSE here.
            loss = rmse_loss 
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_nasa_score += nasa_score.item()
        
        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] | LR: {scheduler.get_last_lr()[0]:.6f} | RMSE Loss: {epoch_loss/len(dataloader):.2f} | NASA Score: {epoch_nasa_score:.2f}")

    print("Training complete. ")
    return model

if __name__ == "__main__":
    import os
    if os.path.exists("train_FD002.txt"):
        model = train_model()
        torch.save(model.state_dict(), "fusion_model.pth")
        print("Model saved to 'fusion_model.pth'. You can now run 'python predict.py'.")
    else:
        print("Dataset 'train_FD002.txt' not found in the current directory.")
