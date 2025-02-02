import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
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


# Enhanced Feature Engineering
def create_features(df, is_training=True):
    # Create lag features
    for lag in range(1, 8):  # Increased lag window
        if is_training:
            df[f'total_cases_lag_{lag}'] = df['total_cases'].shift(lag)
        else:
            df[f'total_cases_lag_{lag}'] = df_train['total_cases'].mean()

    # Create rolling mean features
    if is_training:
        df['rolling_mean_7'] = df['total_cases'].rolling(window=7, min_periods=1).mean()
        df['rolling_mean_14'] = df['total_cases'].rolling(window=14, min_periods=1).mean()
        df['rolling_std_7'] = df['total_cases'].rolling(window=7, min_periods=1).std()
    else:
        df['rolling_mean_7'] = df_train['total_cases'].mean()
        df['rolling_mean_14'] = df_train['total_cases'].mean()
        df['rolling_std_7'] = df_train['total_cases'].std()

    # Create cyclical features for week of year
    if 'weekofyear' in df.columns:
        df['weekofyear_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52.0)
        df['weekofyear_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52.0)

    return df


# Apply feature engineering
df_train = create_features(df_train, is_training=True)
df_test = create_features(df_test, is_training=False)

# Drop rows with NaN values in training set
df_train.dropna(inplace=True)

# Prepare features
feature_columns = [col for col in df_train.columns if col != 'total_cases']
X = df_train[feature_columns]
y = df_train['total_cases']

# Use RobustScaler instead of MinMaxScaler to handle outliers better
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for LSTM
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split with stratification based on quartiles
y_quartiles = pd.qcut(y, q=4, labels=False)
X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y, test_size=0.2, random_state=42, stratify=y_quartiles)


# Build improved model
def build_model(input_shape):
    model = Sequential([
        # First LSTM layer with L2 regularization
        LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True,
             kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),

        # Second LSTM layer
        LSTM(64, activation='relu', return_sequences=True,
             kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),

        # Third LSTM layer
        LSTM(32, activation='relu',
             kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),

        # Dense layers
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dense(1, activation='relu')  # Using ReLU to ensure non-negative predictions
    ])
    return model


# Initialize model
model = build_model((X_train.shape[1], X_train.shape[2]))

# Compile with custom learning rate
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])  # Using Huber loss for robustness

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

# Train with improved settings
history = model.fit(
    X_train,
    y_train,
    epochs=150,  # Increased epochs since we have early stopping
    batch_size=16,  # Smaller batch size for better generalization
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate on validation set
y_pred = model.predict(X_val)
mae = np.mean(np.abs(y_val - y_pred.flatten()))
print(f"\nValidation MAE: {mae:.4f}")

# Prepare test data
X_test = df_test[feature_columns]
X_test_scaled = scaler.transform(X_test)
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Generate predictions
test_predictions = model.predict(X_test_reshaped)
test_predictions = np.maximum(test_predictions, 0)  # Ensure non-negative
test_predictions = np.round(test_predictions)  # Round to nearest integer

# Save predictions
submission_df = pd.DataFrame({
    'total_cases': test_predictions.flatten().astype(int)
})
submission_df.to_csv(r"C:\Users\prath\PycharmProjects\CodeSocHackathon\DiseaseSpreadPrediction\Data\submission.csv",
                     index=False)
print("Predictions saved to submission.csv")

# Print model summary and final training metrics
print("\nModel Summary:")
model.summary()