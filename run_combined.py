import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import os
import matplotlib.pyplot as plt

# --- Configuration (For All Combined Datasets) ---
MODEL_PATH = 'lstm_autoencoder_model_combined.h5' # New model file for the combined dataset
SEQ_LENGTH = 50
SENSOR_COLS = [f's{i}' for i in range(1, 22)]
DATASET_IDS = ['FD001', 'FD002', 'FD003', 'FD004']

# --- STEP 1: Load and Combine All Training Data ---
def step_1_load_all_data():
    print("--- Step 1: Loading and Combining All Training Data ---")
    all_train_df = []
    col_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + SENSOR_COLS
    
    for ds_id in DATASET_IDS:
        train_path = f'data/train_{ds_id}.txt'
        df_train = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
        # Add a dataset identifier to each engine_id to make them unique across datasets
        df_train['engine_id'] = df_train['engine_id'].apply(lambda x: f"{ds_id}_{x}")
        all_train_df.append(df_train)
        print(f"Loaded {train_path}")

    combined_df = pd.concat(all_train_df, ignore_index=True)
    print(f"\nAll training data combined. Total rows: {combined_df.shape[0]}")
    print("-" * 50 + "\n")
    return combined_df

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

# --- STEP 3: Build, Train, and Save the Combined Model ---
def step_3_build_and_train(X_train):
    print("--- Step 3: Building and Training Combined 'Master' Model ---")
    num_features = X_train.shape[2]
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(SEQ_LENGTH, num_features), return_sequences=False),
        RepeatVector(SEQ_LENGTH),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(num_features))
    ])
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    
    print("\nTraining combined model... (This will take a significant amount of time)")
    # Increased batch size for the larger dataset to make training more efficient
    model.fit(X_train, X_train, epochs=10, batch_size=256, verbose=1) 
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        
    model.save(MODEL_PATH)
    print(f"Model trained and saved to '{MODEL_PATH}'")
    print("-" * 50 + "\n")
    return model

# --- STEP 4: Evaluate the Combined Model on Each Test Set ---
def step_4_evaluate(model, scaler):
    print("--- Step 4: Evaluating Combined Model on Each Test Set ---")

    for ds_id in DATASET_IDS:
        print(f"\n🔍 Evaluating on {ds_id}...")
        test_path = f'data/test_{ds_id}.txt'
        rul_path = f'data/RUL_{ds_id}.txt'
        output_plot_path = f'plots/evaluation_plot_{ds_id}_master_model.png'

        col_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + SENSOR_COLS
        df_test = pd.read_csv(test_path, sep=r'\s+', header=None, names=col_names)

        # Load RUL values
        with open(rul_path, 'r') as f:
            rul_true = [int(line.strip()) for line in f if line.strip()]
        print(f"👉 RUL entries: {len(rul_true)}")

        # Normalize sensor data
        df_test[SENSOR_COLS] = scaler.transform(df_test[SENSOR_COLS])

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

        if len(X_test) != len(valid_rul_values):
            print(f"⚠️ WARNING: X_test ({len(X_test)}) and RUL ({len(valid_rul_values)}) length mismatch.")
            min_len = min(len(X_test), len(valid_rul_values))
            X_test = X_test[:min_len]
            valid_rul_values = valid_rul_values[:min_len]

        test_preds = model.predict(X_test, verbose=0)
        anomaly_scores = np.mean(np.abs(test_preds - X_test), axis=(1, 2)).flatten()

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_rul_values, anomaly_scores, alpha=0.6, color='purple', label='Anomaly Score')
        plt.title(f'Master Model Evaluation on {ds_id}')
        plt.xlabel('True Remaining Useful Life (Cycles)')
        plt.ylabel('Anomaly Score (Reconstruction Error)')
        plt.grid(True)
        plt.legend()

        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
        plt.savefig(output_plot_path)
        plt.close()
        print(f"✅ Plot saved: '{output_plot_path}'")

    print("\n✅ Evaluation complete for all datasets.")
    print("-" * 50 + "\n")



# --- Main execution block ---
if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        df_train_loaded = step_1_load_all_data()
        X_train_processed, data_scaler = step_2_preprocess(df_train_loaded)
        trained_model = step_3_build_and_train(X_train_processed)
    else:
        print(f"--- Master model already exists at '{MODEL_PATH}', loading it. ---")
        # We still need the scaler, which must be fit on the combined training data
        df_train_loaded = step_1_load_all_data()
        _, data_scaler = step_2_preprocess(df_train_loaded)
        trained_model = load_model(MODEL_PATH, compile=False)
        trained_model.compile(optimizer='adam', loss='mae')
        print("Model loaded successfully.")
        print("-" * 50 + "\n")
        
    step_4_evaluate(trained_model, data_scaler)

    print("All steps complete. Check your 'plots' folder for the new evaluation plots.")
