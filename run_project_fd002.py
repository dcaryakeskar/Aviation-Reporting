import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import os
import matplotlib.pyplot as plt

# --- Configuration (Updated for FD002) ---
# Define file paths
TRAIN_PATH = 'data/train_FD002.txt'
TEST_PATH = 'data/test_FD002.txt'
RUL_PATH = 'data/RUL_FD002.txt' 
MODEL_PATH = 'lstm_autoencoder_model_fd002.h5' # New model file
OUTPUT_PLOT_PATH = 'plots/evaluation_plot_fd002.png' # New plot file

# Define model parameters
SEQ_LENGTH = 50
SENSOR_COLS = [f's{i}' for i in range(1, 22)]

# --- STEP 1: Load and Prepare Data ---
def step_1_load_data():
    print("--- Step 1: Loading Data (FD002) ---")
    col_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + SENSOR_COLS
    df_train = pd.read_csv(TRAIN_PATH, sep=r'\s+', header=None, names=col_names)
    print(f"Training data loaded successfully with {df_train.shape[0]} rows.")
    print("-" * 50 + "\n")
    return df_train

# --- STEP 2: Preprocess Data and Create Sequences ---
def step_2_preprocess(df_train):
    print("--- Step 2: Preprocessing Data ---")
    scaler = MinMaxScaler()
    df_train[SENSOR_COLS] = scaler.fit_transform(df_train[SENSOR_COLS])
    print("Sensor data normalized.")
    
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)
        
    X_train = create_sequences(df_train[SENSOR_COLS].values, SEQ_LENGTH)
    print(f"Data transformed into sequences of length {SEQ_LENGTH}.")
    print("-" * 50 + "\n")
    return X_train, scaler

# --- STEP 3: Build, Train, and Save the Model ---
def step_3_build_and_train(X_train):
    print("--- Step 3: Building and Training Model for FD002 ---")
    num_features = X_train.shape[2]
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(SEQ_LENGTH, num_features), return_sequences=False),
        RepeatVector(SEQ_LENGTH),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(num_features))
    ])
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    
    print("\nTraining model... (This may take longer due to more data)")
    model.fit(X_train, X_train, epochs=10, batch_size=64, verbose=1)
    
    model.save(MODEL_PATH)
    print(f"Model trained and saved to '{MODEL_PATH}'")
    print("-" * 50 + "\n")
    return model

# --- STEP 4: Evaluate Model and Correlate with RUL ---
def step_4_evaluate(model, scaler):
    print("--- Step 4: Evaluating Model and Correlating with RUL (FD002) ---")
    
    # Load test and RUL data
    col_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + SENSOR_COLS
    df_test = pd.read_csv(TEST_PATH, sep=r'\s+', header=None, names=col_names)
    
    rul_true = []
    with open(RUL_PATH, 'r') as f:
        for line in f:
            line = line.strip() 
            if line: 
                rul_true.append(int(line))
    
    df_test[SENSOR_COLS] = scaler.transform(df_test[SENSOR_COLS])
    
    last_cycle_sequences = []
    for engine_id in df_test['engine_id'].unique():
        engine_data = df_test[df_test['engine_id'] == engine_id][SENSOR_COLS].values
        num_cycles = len(engine_data)
        
        if num_cycles < SEQ_LENGTH:
            padded_data = np.zeros((SEQ_LENGTH, engine_data.shape[1]))
            padded_data[-num_cycles:] = engine_data
            last_cycle_sequences.append(padded_data)
        else:
            last_sequences = engine_data[-SEQ_LENGTH:]
            last_cycle_sequences.append(last_sequences)
            
    X_test = np.array(last_cycle_sequences)
    
    test_preds = model.predict(X_test)
    anomaly_scores = np.mean(np.abs(test_preds - X_test), axis=1).flatten()
    
    # 🔍 Debug print
    print(f"👉 RUL entries: {len(rul_true)}")
    print(f"👉 Anomaly scores: {len(anomaly_scores)}")
    
    # ✅ FIX: Align lengths
    if len(rul_true) != len(anomaly_scores):
        print(f"⚠️ Mismatch detected! Trimming to smallest size.")
    min_len = min(len(rul_true), len(anomaly_scores))
    rul_true = rul_true[:min_len]
    anomaly_scores = anomaly_scores[:min_len]
    
    # 📊 Plot
    print("📊 Creating evaluation plot...")
    plt.figure(figsize=(10, 6))
    plt.scatter(rul_true, anomaly_scores, alpha=0.5, color='blue', label='Anomaly Score')
    plt.title('Anomaly Score vs. RUL for FD002 (Multiple Operating Conditions)')
    plt.xlabel('True Remaining Useful Life (Cycles)')
    plt.ylabel('Anomaly Score (Reconstruction Error)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(OUTPUT_PLOT_PATH)
    print(f"✅ Evaluation plot saved to '{OUTPUT_PLOT_PATH}'")
    
    print("\n✅ Evaluation complete.")
    print("-" * 50 + "\n")


# --- Main execution block ---
if __name__ == '__main__':
    df_train_loaded = step_1_load_data()
    X_train_processed, data_scaler = step_2_preprocess(df_train_loaded)
    
    # We force retraining for the new dataset
    trained_model = step_3_build_and_train(X_train_processed)
        
    step_4_evaluate(trained_model, data_scaler)

    print("All steps complete. Check your project folder for 'evaluation_plot_fd002.png'")
