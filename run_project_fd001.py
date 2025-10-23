import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# =========================
# CONFIGURATION
# =========================
TRAIN_PATH = 'data/train_FD001.txt'
TEST_PATH = 'data/test_FD001.txt'
RUL_PATH = 'data/RUL_FD001.txt'          # File containing Remaining Useful Life values
MODEL_PATH = 'lstm_autoencoder_model.h5'
OUTPUT_PLOT_PATH = 'evaluation_plot.png' # Where to save the plot

SEQ_LENGTH = 50
SENSOR_COLS = [f's{i}' for i in range(1, 22)]  # s1 to s21

# =========================
# STEP 1 – LOAD DATA
# =========================
def step_1_load_data():
    """Loads the training dataset."""
    print("--- Step 1: Loading Data ---")

    col_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + SENSOR_COLS
    df_train = pd.read_csv(TRAIN_PATH, sep=r'\s+', header=None, names=col_names)

    print(f"✅ Training data loaded successfully with {df_train.shape[0]} rows.")
    print("-" * 50 + "\n")
    return df_train

# =========================
# STEP 2 – PREPROCESS DATA
# =========================
def step_2_preprocess(df_train):
    """Normalizes sensor data and converts it into sequential windows for training."""
    print("--- Step 2: Preprocessing Data ---")

    scaler = MinMaxScaler()
    df_train[SENSOR_COLS] = scaler.fit_transform(df_train[SENSOR_COLS])
    print("✅ Sensor data normalized.")

    # ---- Create sequential training samples ----
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)

    X_train = create_sequences(df_train[SENSOR_COLS].values, SEQ_LENGTH)

    print(f"✅ Data transformed into sequences of length {SEQ_LENGTH}.")
    print("-" * 50 + "\n")
    return X_train, scaler

# =========================
# STEP 3 – BUILD & TRAIN MODEL
# =========================
def step_3_build_and_train(X_train):
    """Builds, trains, and saves the LSTM Autoencoder."""
    print("--- Step 3: Building and Training Model ---")

    num_features = X_train.shape[2]

    # ---- LSTM Autoencoder ----
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(SEQ_LENGTH, num_features), return_sequences=False),
        RepeatVector(SEQ_LENGTH),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(num_features))
    ])
    model.compile(optimizer='adam', loss='mae')

    model.summary()

    print("\n🚀 Training model (this may take a few minutes)...")
    model.fit(X_train, X_train, epochs=10, batch_size=64, verbose=1)

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"✅ Model trained and saved to '{MODEL_PATH}'")
    print("-" * 50 + "\n")
    return model

# =========================
# STEP 4 – EVALUATE MODEL
# =========================
def step_4_evaluate(model, scaler):
    print("--- Step 4: Evaluating Model and Correlating with RUL ---")
    
    # Load test data
    col_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + SENSOR_COLS
    df_test = pd.read_csv(TEST_PATH, sep=r'\s+', header=None, names=col_names)
    
    # Load RUL data manually
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
        if len(engine_data) < SEQ_LENGTH:
            padded_data = np.zeros((SEQ_LENGTH, engine_data.shape[1]))
            padded_data[-len(engine_data):] = engine_data
            last_cycle_sequences.append(padded_data)
        else:
            last_cycle_sequences.append(engine_data[-SEQ_LENGTH:])
    
    X_test = np.array(last_cycle_sequences)
    
    test_preds = model.predict(X_test)
    anomaly_scores = np.mean(np.abs(test_preds - X_test), axis=1).flatten()

    # 🔥 FIX: Align lengths before plotting
    if len(rul_true) != len(anomaly_scores):
        print(f"⚠️ Mismatch detected! RUL: {len(rul_true)}, Scores: {len(anomaly_scores)}")
    min_len = min(len(rul_true), len(anomaly_scores))
    rul_true = rul_true[:min_len]
    anomaly_scores = anomaly_scores[:min_len]
    
    # Plot
    print("📊 Creating evaluation plot...")
    plt.figure(figsize=(10, 6))
    plt.scatter(rul_true, anomaly_scores, alpha=0.5, color='blue', label='Anomaly Score')
    plt.title('Anomaly Score vs. Remaining Useful Life (RUL)')
    plt.xlabel('True Remaining Useful Life (Cycles)')
    plt.ylabel('Anomaly Score (Reconstruction Error)')
    plt.grid(True)
    plt.legend()
    plt.savefig(OUTPUT_PLOT_PATH)
    print(f"✅ Evaluation plot saved to '{OUTPUT_PLOT_PATH}'")
    print("\n✅ Evaluation complete.")
    print("-" * 50 + "\n")


# =========================
# MAIN EXECUTION BLOCK
# =========================
if __name__ == '__main__':
    # Step 1: Load training data
    df_train_loaded = step_1_load_data()

    # Step 2: Preprocess & create sequences
    X_train_processed, data_scaler = step_2_preprocess(df_train_loaded)

    # Step 3: Build or load model
    if not os.path.exists(MODEL_PATH):
        trained_model = step_3_build_and_train(X_train_processed)
    else:
        print(f"--- Step 3: Skipped (Model already exists at '{MODEL_PATH}') ---")
        print("📥 Loading existing model...")
        trained_model = load_model(MODEL_PATH, compile=False)
        trained_model.compile(optimizer='adam', loss='mae')
        print("✅ Model loaded successfully.")
        print("-" * 50 + "\n")

    # Step 4: Evaluate the model
    step_4_evaluate(trained_model, data_scaler)

    print("🎯 All steps complete. Check your project folder for 'evaluation_plot.png'.")
    print("👉 You can still run the dashboard with: streamlit run app.py")
