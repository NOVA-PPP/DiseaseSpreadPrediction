import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Load datasets
df_train = pd.read_csv(
    r"C:\Users\prath\PycharmProjects\CodeSocHackathon\DiseaseSpreadPrediction\Data\dengue_train_cleaned.csv")
df_test = pd.read_csv(
    r"C:\Users\prath\PycharmProjects\CodeSocHackathon\DiseaseSpreadPrediction\Data\dengue_test_cleaned.csv")


def create_features(df, is_training=True):
    # Essential lag features (focus on recent history)
    if is_training:
        for lag in range(1, 5):
            df[f'total_cases_lag_{lag}'] = df['total_cases'].shift(lag)

        # Exponential moving averages (give more weight to recent data)
        df['ema_3'] = df['total_cases'].ewm(span=3, adjust=False).mean()
        df['ema_7'] = df['total_cases'].ewm(span=7, adjust=False).mean()

        # Rolling statistics with shorter windows
        df['rolling_mean_3'] = df['total_cases'].rolling(window=3, min_periods=1).mean()
        df['rolling_std_3'] = df['total_cases'].rolling(window=3, min_periods=1).std()

        # Safe rate of change calculation
        df['rate_of_change'] = df['total_cases'].pct_change()
        df['rate_of_change'] = df['rate_of_change'].replace([np.inf, -np.inf], np.nan)
        df['rate_of_change'] = df['rate_of_change'].fillna(0)  # Fill NaN with 0
    else:
        # Initialize with training data statistics for test set
        train_mean = df_train['total_cases'].mean()
        train_std = df_train['total_cases'].std()
        for lag in range(1, 5):
            df[f'total_cases_lag_{lag}'] = train_mean
        df['ema_3'] = train_mean
        df['ema_7'] = train_mean
        df['rolling_mean_3'] = train_mean
        df['rolling_std_3'] = train_std
        df['rate_of_change'] = 0

    # Cyclical encoding of weekofyear (if present)
    if 'weekofyear' in df.columns:
        df['weekofyear_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52.0)
        df['weekofyear_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52.0)

    return df


# Apply feature engineering
df_train = create_features(df_train, is_training=True)
df_test = create_features(df_test, is_training=False)

# Drop rows with NaN values
df_train.dropna(inplace=True)

# Prepare features
feature_columns = [col for col in df_train.columns if col != 'total_cases']
X = df_train[feature_columns]
y = df_train['total_cases']

# Replace any remaining infinities with 0
X = X.replace([np.inf, -np.inf], 0)

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split data with stratification
y_quartiles = pd.qcut(y, q=4, labels=False)
X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y, test_size=0.15, random_state=42, stratify=y_quartiles)


def build_optimized_model(input_shape):
    model = Sequential([
        # First LSTM layer - reduced units, lighter regularization
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True,
             kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),

        # Second LSTM layer
        LSTM(32, activation='relu',
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),

        # Dense layers with reduced complexity
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dense(8, activation='relu'),
        Dense(1, activation='relu')
    ])
    return model


# Initialize model
model = build_optimized_model((X_train.shape[1], X_train.shape[2]))

# Custom learning rate and optimizer
optimizer = Adam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Refined callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=0.0005,
    verbose=1
)

# Train with optimized settings
history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=24,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate
y_pred = model.predict(X_val)
mae = np.mean(np.abs(y_val - y_pred.flatten()))
print(f"\nValidation MAE: {mae:.4f}")

# Generate test predictions
X_test = df_test[feature_columns]
X_test = X_test.replace([np.inf, -np.inf], 0)  # Handle infinities in test data
X_test_scaled = scaler.transform(X_test)
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
test_predictions = model.predict(X_test_reshaped)
test_predictions = np.maximum(test_predictions, 0)
test_predictions = np.round(test_predictions)

# Save predictions
submission_df = pd.DataFrame({
    'total_cases': test_predictions.flatten().astype(int)
})
submission_df.to_csv(r"C:\Users\prath\PycharmProjects\CodeSocHackathon\DiseaseSpreadPrediction\Data\submission.csv",
                     index=False)
print("Predictions saved to submission.csv")