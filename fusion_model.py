import torch
import torch.nn as nn
import torch.nn.functional as F

class SensorEncoder(nn.Module):
    """
    Encoder for a single sensor modality.
    Takes sliding window time-series data and extracts a feature vector.
    """
    def __init__(self, input_dim=1, hidden_dim=64, feature_dim=32):
        super(SensorEncoder, self).__init__()
        # Use a simple 1D CNN followed by Global Average Pooling
        # Alternatively, an LSTM could be used here.
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=feature_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, input_dim, sequence_length)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # Global Average Pooling over the sequence dimension
        x = torch.mean(x, dim=-1)
        # Output shape: (batch_size, feature_dim)
        return x

class ReliabilityEstimator(nn.Module):
    """
    Estimates the reliability (trustworthiness) of a sensor based on its extracted features.
    """
    def __init__(self, feature_dim=32, hidden_dim=16):
        super(ReliabilityEstimator, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        # No sigmoid here entirely; will apply softmax over all sensor reliabilities later
        # OR we can apply sigmoid if we want independent reliability > threshold. 
        # For typical weighted fusion, we output raw logits and softmax them across sensors.
        
    def forward(self, features):
        # features shape: (batch_size, feature_dim)
        x = self.relu(self.fc1(features))
        logits = self.fc2(x)
        # Output shape: (batch_size, 1)
        return logits

class ReliabilityWeightedFusion(nn.Module):
    """
    The full multimodal fusion architecture.
    """
    def __init__(self, num_sensors=14, sequence_length=30, feature_dim=32):
        super(ReliabilityWeightedFusion, self).__init__()
        self.num_sensors = num_sensors
        
        # Create independent encoders and reliability estimators for each sensor
        self.encoders = nn.ModuleList([
            SensorEncoder(input_dim=1, feature_dim=feature_dim) for _ in range(num_sensors)
        ])
        
        self.reliability_estimators = nn.ModuleList([
            ReliabilityEstimator(feature_dim=feature_dim) for _ in range(num_sensors)
        ])
        
        # Final predictor network taking the fused feature vector
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Outputs RUL (Remaining Useful Life)
        )
        
    def forward(self, sensor_inputs):
        """
        sensor_inputs: list of length `num_sensors`, each element is a tensor 
                       of shape (batch_size, sequence_length).
        """
        batch_size = sensor_inputs[0].shape[0]
        
        feature_vectors = []
        reliability_logits = []
        
        for i in range(self.num_sensors):
            # Add channel dimension: (batch_size, 1, sequence_length)
            x_i = sensor_inputs[i].unsqueeze(1)
            
            # 1. Encode sensor data to feature vector
            h_i = self.encoders[i](x_i)
            feature_vectors.append(h_i)
            
            # 2. Estimate reliability logit
            r_logit_i = self.reliability_estimators[i](h_i)
            reliability_logits.append(r_logit_i)
            
        # Stack features: (batch_size, num_sensors, feature_dim)
        stacked_features = torch.stack(feature_vectors, dim=1)
        
        # Stack reliability logits: (batch_size, num_sensors, 1)
        stacked_logits = torch.stack(reliability_logits, dim=1)
        
        # 3. Calculate fusion weights via Softmax across the sensor dimension
        # Shape: (batch_size, num_sensors, 1)
        weights = F.softmax(stacked_logits, dim=1)
        
        # 4. Weighted fusion
        # Multiply features by their weights and sum across sensors
        # Shape after weighting & sum: (batch_size, feature_dim)
        fused_vector = torch.sum(weights * stacked_features, dim=1)
        
        # 5. Predict final Final RUL
        rul_prediction = self.predictor(fused_vector)
        
        return rul_prediction, weights

if __name__ == "__main__":
    # Example usage:
    batch_size = 16
    seq_len = 30
    
    # Simulate 14 sensor inputs
    inputs = [torch.randn(batch_size, seq_len) for _ in range(14)]
    
    model = ReliabilityWeightedFusion(num_sensors=14, sequence_length=seq_len)
    predicted_rul, fusion_weights = model(inputs)
    
    print("Predicted RUL shape:", predicted_rul.shape)
    print("Fusion Weights shape:", fusion_weights.shape)
    # The sum of weights across dimension 1 for any batch element should be ~1.0
    print("Sum of weights:", torch.sum(fusion_weights[0, :, 0]).item())
