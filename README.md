## Name : Kishore S
## Reg No : 212222240050
# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load and clean the CSV file
data = pd.read_csv("/content/NFLX.csv")  # Adjust path if necessary
data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')  # Clean 'Close' prices
data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')  # Clean 'Volume'

# Set 'Date' as index
data.set_index('Date', inplace=True)

# Perform Augmented Dickey-Fuller (ADF) test for stationarity
result = adfuller(data['Close'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into training and testing sets (80% training, 20% testing)
train_data = data.iloc[:int(0.8*len(data))]
test_data = data.iloc[int(0.8*len(data)):]

# Fit an AutoRegressive (AR) model with 13 lags on the training data
lag_order = 13
model = AutoReg(train_data['Close'], lags=lag_order)
model_fit = model.fit()

# Plot Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
plot_acf(data['Close'].dropna())
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(data['Close'].dropna())
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Make predictions using the AR model
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

# Compare the predictions with the test data
mse = mean_squared_error(test_data['Close'], predictions)
print('Mean Squared Error (MSE):', mse)

# Plot the test data and predictions
plt.plot(test_data.index, test_data['Close'], label='Test Data')
plt.plot(test_data.index, predictions, label='Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price (Close)')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.show()

```
### OUTPUT:

GIVEN DATA

![image](https://github.com/user-attachments/assets/0e6119a1-e2c8-409c-afdd-65a7d3bcc8d5)


PACF - ACF

![image](https://github.com/user-attachments/assets/273e5666-8446-43ba-a8f9-0bf6b4fcd69b)

![image](https://github.com/user-attachments/assets/35c87005-ec17-4197-85fa-b3e845344e28)


PREDICTION

![image](https://github.com/user-attachments/assets/1ccb8e83-daeb-47d3-8e90-87ca800d010c)


FINIAL PREDICTION

![image](https://github.com/user-attachments/assets/6429da26-1583-42fa-b4bb-d6711f6dfe7c)


### RESULT:
Thus the implementation of auto regression function using python was successfully completed.
