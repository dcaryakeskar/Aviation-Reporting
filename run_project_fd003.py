import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# --- Configuration ---
TRAIN_PATH = 'data/train_FD003.txt'
TEST_PATH = 'data/test_FD003.txt'
RUL_PATH = 'data/RUL_FD003.txt'
MODEL_PATH = 'lstm_autoencoder_model_fd003.keras'  # Modern format
OUTPUT_PLOT_PATH = 'plots/evaluation_plot_fd003.png'
SEQ_LENGTH = 50
SENSOR_COLS = [f's{i}' for i in range(1, 22)]


# --- STEP 1: Load Training Data ---
def step_1_load_data():
    print("--- Step 1: Loading Data (FD003) ---")
    col_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + SENSOR_COLS
    df = pd.read_csv(TRAIN_PATH, sep=r'\s+', header=None, names=col_names)
    print(f"✅ Loaded training data with {df.shape[0]} rows.")
    print("-" * 50 + "\n")
    return df


# --- STEP 2: Preprocess and Create Sequences ---
def step_2_preprocess(df_train):
    print("--- Step 2: Preprocessing Data ---")
    scaler = MinMaxScaler()
    df_train[SENSOR_COLS] = scaler.fit_transform(df_train[SENSOR_COLS])
    print("✅ Sensor data normalized.")

    def create_sequences(data, seq_len):
        return np.array([data[i:i + seq_len] for i in range(len(data) - seq_len + 1)])

    X_train = create_sequences(df_train[SENSOR_COLS].values, SEQ_LENGTH)
    print(f"✅ Created {X_train.shape[0]} sequences of shape {X_train.shape[1:]}")
    print("-" * 50 + "\n")
    return X_train, scaler


# --- STEP 3: Build, Train and Save Model ---
def step_3_build_and_train(X_train):
    print("--- Step 3: Building and Training Model ---")
    num_features = X_train.shape[2]
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(SEQ_LENGTH, num_features), return_sequences=False),
        RepeatVector(SEQ_LENGTH),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(num_features))
    ])
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    print("\n📚 Training model...")
    model.fit(X_train, X_train, epochs=10, batch_size=64, verbose=1)

    model.save(MODEL_PATH)
    print(f"✅ Model trained and saved to '{MODEL_PATH}'")
    print("-" * 50 + "\n")
    return model


# --- STEP 4: Evaluate and Plot ---
def step_4_evaluate(model, scaler):
    print("--- Step 4: Evaluating Model (FD003) ---")
    col_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + SENSOR_COLS
    df_test = pd.read_csv(TEST_PATH, sep=r'\s+', header=None, names=col_names)

    # Load RUL values
    with open(RUL_PATH, 'r') as f:
        rul_true = [int(line.strip()) for line in f if line.strip()]

    print(f"👉 RUL entries: {len(rul_true)}")
    df_test[SENSOR_COLS] = scaler.transform(df_test[SENSOR_COLS])

    # Prepare last-cycle sequences for each engine
    last_cycle_sequences = []
    for engine_id in df_test['engine_id'].unique():
        engine_data = df_test[df_test['engine_id'] == engine_id][SENSOR_COLS].values
        if len(engine_data) < SEQ_LENGTH:
            padded = np.zeros((SEQ_LENGTH, engine_data.shape[1]))
            padded[-len(engine_data):] = engine_data
            last_cycle_sequences.append(padded)
        else:
            last_cycle_sequences.append(engine_data[-SEQ_LENGTH:])

    X_test = np.array(last_cycle_sequences)
    print(f"👉 Test sequences: {len(X_test)}")

    if len(X_test) != len(rul_true):
        print(f"❌ ERROR: Test and RUL length mismatch. Aborting.")
        return

    # Predict and compute anomaly scores
    test_preds = model.predict(X_test)
    anomaly_scores = np.mean(np.abs(test_preds - X_test), axis=(1, 2))  # ✅ FIXED here

    # Create plots folder if not exists
    os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)

    print("📊 Creating evaluation plot...")
    plt.figure(figsize=(10, 6))
    plt.scatter(rul_true, anomaly_scores, alpha=0.6, color='purple', label='Anomaly Score')
    plt.title('Anomaly Score vs. RUL for FD003 (Two Fault Modes)')
    plt.xlabel('True Remaining Useful Life (Cycles)')
    plt.ylabel('Anomaly Score (Reconstruction Error)')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_PLOT_PATH)
    print(f"✅ Evaluation plot saved to '{OUTPUT_PLOT_PATH}'")
    print("-" * 50 + "\n")


# --- MAIN ---
if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        df_train = step_1_load_data()
        X_train, scaler = step_2_preprocess(df_train)
        model = step_3_build_and_train(X_train)
    else:
        print(f"📦 Found existing model at '{MODEL_PATH}'. Loading...")
        df_train = step_1_load_data()
        _, scaler = step_2_preprocess(df_train)
        model = load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='mae')
        print("✅ Model loaded.")
        print("-" * 50 + "\n")

    step_4_evaluate(model, scaler)
    print("🎉 All steps complete. Check your 'plots' folder for the evaluation plot.")
