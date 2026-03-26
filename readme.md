# MTE546 Reliability Weighted Neural Network

### Data preprocessing

* Normalizatoin based on operating conditions. Data is only normalized against data that is in the same operating conditions because readings at 40,000 ft are different than at sea level


## Neural Network Architecture

CNNs
Time series data of sensor is passed into a CNN and turned into a feature vector.


Reliability Network
Network generates a trust score for every sensor

Each sensor is then multiplied by its weight. Low reliability sensors are essentially dropped and only the high reliability remain.

Final MLP
The output of the cleaned sensors is passed into a final neural network to output a RUL

## Evaluation
RMSE and NASA score
