import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset (ensure to replace with the correct path to the dataset)
train_df = pd.read_csv('rossmann_store_sales.csv')

#  Preprocessing
#  Handle Missing Values
train_df['CompetitionDistance'].fillna(train_df['CompetitionDistance'].median(), inplace=True)
train_df['Promo2SinceYear'].fillna(0, inplace=True)

#  Feature Engineering
# Convert 'Date' to datetime format
train_df['Date'] = pd.to_datetime(train_df['Date'])

# Extract Weekday, Weekend, and Day of the Month
train_df['Weekday'] = train_df['Date'].dt.weekday
train_df['Weekend'] = train_df['Date'].dt.weekday >= 5
train_df['DayOfMonth'] = train_df['Date'].dt.day

# Calculate Days to and Days after Holidays (StateHoliday)
train_df['DaysToHoliday'] = train_df['Date'].apply(lambda x: (x - pd.Timestamp('2025-12-25')).days)  # Example: Christmas
train_df['DaysAfterHoliday'] = train_df['DaysToHoliday'].apply(lambda x: max(x, 0))  # Only positive values

# Beginning, Mid, and End of the Month
train_df['MonthStart'] = train_df['Date'].dt.is_month_start.astype(int)
train_df['MonthEnd'] = train_df['Date'].dt.is_month_end.astype(int)

#  Scaling the Data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(train_df[['Sales', 'Customers', 'CompetitionDistance']])
train_df[['Sales', 'Customers', 'CompetitionDistance']] = scaled_features

#  Encoding Categorical Features
train_df = pd.get_dummies(train_df, columns=['StoreType', 'Assortment', 'StateHoliday'])

#  Preparing Target Variable and Features
X = train_df.drop(['Sales', 'Date'], axis=1)
y = train_df['Sales']

#  Building Models with Sklearn Pipelines
#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Define the Random Forest Model in a Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))  # Random Forest model
])

#  Fit the Random Forest Model
pipeline.fit(X_train, y_train)

#  Predict Sales with the Model
y_pred = pipeline.predict(X_test)

#  Evaluate the Model Performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Random Forest - Mean Squared Error: {mse}")
print(f"Random Forest - Mean Absolute Error: {mae}")

#  Serialize the Model
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
filename = f"random_forest_model_{timestamp}.pkl"
joblib.dump(pipeline, filename)

#  Feature Importance from Random Forest Model
importances = pipeline.named_steps['rf'].feature_importances_
features = X.columns

plt.barh(features, importances)
plt.title("Random Forest Feature Importances")
plt.show()

# Deep Learning Model: LSTM
# Preparing Time Series Data for LSTM
def create_dataset(df, time_step=1):
    X, y = [], []
    for i in range(len(df) - time_step):
        X.append(df.iloc[i:(i + time_step)].values)
        y.append(df.iloc[i + time_step]['Sales'])
    return np.array(X), np.array(y)

# Create the time-series dataset (for LSTM)
time_step = 30  # Use 30 days of data to predict the next day's sales
X_lstm, y_lstm = create_dataset(train_df[['Sales']], time_step)

#  Scale the LSTM Data
scaler_lstm = StandardScaler()
X_lstm = scaler_lstm.fit_transform(X_lstm.reshape(-1, X_lstm.shape[-1])).reshape(X_lstm.shape)

#  Split the LSTM Data into Training and Test Sets
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

#  Build the LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

#  Train the LSTM Model
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32)

#  Predict with the LSTM Model
y_pred_lstm = lstm_model.predict(X_test_lstm)

#  Evaluate the LSTM Model
mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
print(f"LSTM - Mean Squared Error: {mse_lstm}")

# Serialize the LSTM Model
lstm_model_filename = f"lstm_model_{timestamp}.h5"
lstm_model.save(lstm_model_filename)

# Plot Predictions vs Actual for LSTM
plt.figure(figsize=(10, 6))
plt.plot(y_test_lstm, label='True Sales')
plt.plot(y_pred_lstm, label='Predicted Sales')
plt.title("LSTM Model - Predictions vs Actual Sales")
plt.legend()
plt.show()
