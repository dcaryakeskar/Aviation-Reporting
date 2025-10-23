import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import os
import matplotlib.pyplot as plt

# --- Configuration (FD004) ---
TRAIN_PATH = 'data/train_FD004.txt'
TEST_PATH = 'data/test_FD004.txt'
RUL_PATH = 'data/RUL_FD004.txt'
MODEL_PATH = 'lstm_autoencoder_model_fd004.h5'
OUTPUT_PLOT_PATH = 'plots/evaluation_plot_fd004.png'
SEQ_LENGTH = 50
SENSOR_COLS = [f's{i}' for i in range(1, 22)]

# --- STEP 1: Load Data ---
def step_1_load_data():
    print("--- Step 1: Loading Data (FD004) ---")
    col_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + SENSOR_COLS
    df_train = pd.read_csv(TRAIN_PATH, sep=r'\s+', header=None, names=col_names)
    print(f"✅ Training data loaded successfully with {df_train.shape[0]} rows.")
    print("-" * 50 + "\n")
    return df_train

# --- STEP 2: Preprocess Data ---
def step_2_preprocess(df_train):
    print("--- Step 2: Preprocessing Data ---")
    scaler = MinMaxScaler()
    df_train[SENSOR_COLS] = scaler.fit_transform(df_train[SENSOR_COLS])
    print("✅ Sensor data normalized.")

    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)

    X_train = create_sequences(df_train[SENSOR_COLS].values, SEQ_LENGTH)
    print(f"✅ Data transformed into sequences of length {SEQ_LENGTH}.")
    print("-" * 50 + "\n")
    return X_train, scaler

# --- STEP 3: Build & Train Model ---
def step_3_build_and_train(X_train):
    print("--- Step 3: Building and Training Model for FD004 ---")
    num_features = X_train.shape[2]
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(SEQ_LENGTH, num_features), return_sequences=False),
        RepeatVector(SEQ_LENGTH),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(num_features))
    ])
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    print("\n🚀 Training model...")
    model.fit(X_train, X_train, epochs=10, batch_size=128, verbose=1)
    model.save(MODEL_PATH)
    print(f"✅ Model trained and saved to '{MODEL_PATH}'")
    print("-" * 50 + "\n")
    return model

# --- STEP 4: Evaluate Model + Plot ---
def step_4_evaluate(model, scaler):
    print("--- Step 4: Evaluating Model and Correlating with RUL (FD004) ---")

    col_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + SENSOR_COLS
    df_test = pd.read_csv(TEST_PATH, sep=r'\s+', header=None, names=col_names)

    # Load RUL values
    with open(RUL_PATH, 'r') as f:
        rul_true = [int(line.strip()) for line in f if line.strip()]

    print(f"👉 RUL entries: {len(rul_true)}")

    # Normalize test data
    df_test[SENSOR_COLS] = scaler.transform(df_test[SENSOR_COLS])
    print("✅ Sensor data normalized.")

    last_cycle_sequences = []
    valid_rul_values = []

    for idx, engine_id in enumerate(df_test['engine_id'].unique()):
        engine_data = df_test[df_test['engine_id'] == engine_id][SENSOR_COLS].values
        num_cycles = len(engine_data)

        if num_cycles >= SEQ_LENGTH:
            last_seq = engine_data[-SEQ_LENGTH:]
        else:
            last_seq = np.zeros((SEQ_LENGTH, engine_data.shape[1]))
            last_seq[-num_cycles:] = engine_data

        last_cycle_sequences.append(last_seq)

        if idx < len(rul_true):
            valid_rul_values.append(rul_true[idx])

    X_test = np.array(last_cycle_sequences)

    # Sanity check: match RULs to test sequences
    if len(X_test) != len(valid_rul_values):
        print(f"⚠️ WARNING: X_test ({len(X_test)}) and RUL ({len(valid_rul_values)}) length mismatch.")
        min_len = min(len(X_test), len(valid_rul_values))
        X_test = X_test[:min_len]
        valid_rul_values = valid_rul_values[:min_len]

    # Predict
    test_preds = model.predict(X_test)
    anomaly_scores = np.mean(np.abs(test_preds - X_test), axis=(1, 2))

    print(f"👉 Final aligned counts: RUL = {len(valid_rul_values)}, X_test = {len(X_test)}")
    print("📊 Creating evaluation plot...")

    plt.figure(figsize=(10, 6))
    plt.scatter(valid_rul_values, anomaly_scores, alpha=0.6, color='purple', label='Anomaly Score')
    plt.title('Anomaly Score vs. RUL for FD004 (All Complexities)')
    plt.xlabel('True Remaining Useful Life (Cycles)')
    plt.ylabel('Anomaly Score (Reconstruction Error)')
    plt.grid(True)
    plt.legend()

    os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH)
    print(f"✅ Evaluation plot saved to '{OUTPUT_PLOT_PATH}'")
    print("-" * 50 + "\n")


# --- Main Entry Point ---
if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        df_train_loaded = step_1_load_data()
        X_train_processed, data_scaler = step_2_preprocess(df_train_loaded)
        trained_model = step_3_build_and_train(X_train_processed)
    else:
        print(f"--- Model already exists at '{MODEL_PATH}', loading it. ---")
        df_train_loaded = step_1_load_data()
        _, data_scaler = step_2_preprocess(df_train_loaded)
        trained_model = load_model(MODEL_PATH, compile=False)
        trained_model.compile(optimizer='adam', loss='mae')
        print("✅ Model loaded successfully.")
        print("-" * 50 + "\n")

    step_4_evaluate(trained_model, data_scaler)
    print("🎉 All steps complete. Check your 'plots' folder for the final evaluation plot.")
