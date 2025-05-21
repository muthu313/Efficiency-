import pandas as pd
import numpy as np

# Create a large sample dataset
df = pd.DataFrame({
    'sensor_1': np.random.rand(1000000),
    'sensor_2': np.random.rand(1000000),
    'timestamp': pd.date_range('2023-01-01', periods=1000000, freq='S'),
    'label': np.random.choice([0, 1], size=1000000)
})

# Optimize memory usage
def optimize_dataframe(df):
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

df = optimize_dataframe(df)

# Feature engineering
df['sensor_diff'] = df['sensor_1'] - df['sensor_2']
df['rolling_mean'] = df['sensor_diff'].rolling(window=100).mean()

# Filter valid training data
train_data = df[df['rolling_mean'].notnull()].copy()

# Normalize selected features
for col in ['sensor_1', 'sensor_2', 'sensor_diff', 'rolling_mean']:
    train_data[col] = (train_data[col] - train_data[col].mean()) / train_data[col].std()

# Show accurate output
print(train_data.head())