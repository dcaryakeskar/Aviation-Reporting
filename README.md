# ✈️ AI-Powered Aviation Anomaly Detector  
**Hybrid Deep Learning + Generative AI Diagnostic System**

---

## 🧩 Abstract
This project implements an **AI-driven aviation engine diagnostic platform** that integrates a **sequence-based anomaly detection model** (LSTM Autoencoder) with a **Large Language Model (LLM)** for natural-language diagnosis generation.

The system analyzes multi-sensor turbofan engine data (NASA C-MAPSS datasets) to:
- Detect operational anomalies,
- Identify deviating sensors,
- Generate expert-level technical reports via **Gemini (Google Generative Language API)**,
- Produce fully formatted PDF diagnostic reports.

The application is deployed via **Streamlit**, enabling interactive data exploration and real-time anomaly assessment for both **historical** and **live sensor feeds**.

---

## ⚙️ System Architecture

            ┌────────────────────────────────────────────┐
            │             Streamlit Frontend              │
            │     (Interactive UI + Visualization)       │
            └────────────────────────────────────────────┘
                             │
                             ▼
            ┌────────────────────────────────────────────┐
            │       Data Pipeline & Preprocessing         │
            │  - Load NASA FD001–FD004 datasets          │
            │  - Apply MinMaxScaler (global normalization)│
            └────────────────────────────────────────────┘
                             │
                             ▼
            ┌────────────────────────────────────────────┐
            │        LSTM Autoencoder (Keras)             │
            │  - Sequence length = 50 cycles              │
            │  - 21 sensor inputs                         │
            │  - Trained on normal operating data         │
            │  - Computes reconstruction error (MAE)      │
            └────────────────────────────────────────────┘
                             │
                             ▼
            ┌────────────────────────────────────────────┐
            │   Anomaly Investigation Engine              │
            │  - Identify top 3 deviating sensors         │
            │  - Pass findings to Gemini LLM              │
            │  - Generate human-readable technical report │
            └────────────────────────────────────────────┘
                             │
                             ▼
            ┌────────────────────────────────────────────┐
            │       PDF Report Generator (FPDF)           │
            │  - Summarizes results, plots, LLM analysis  │
            │  - Exports professional-grade report        │
            └────────────────────────────────────────────┘

---

## 🧠 Core Components

### 1. **LSTM Autoencoder**
- **Architecture:**  
  - Encoder: 3 stacked LSTM layers compressing temporal signals  
  - Bottleneck latent representation  
  - Decoder: LSTM layers reconstructing input sequence  
- **Objective Function:**  
  Mean Absolute Error (MAE) between original and reconstructed sensor signals.  
- **Anomaly Score:**  
MAE = (1/n) * Σ |xᵢ - x̂ᵢ|

where:  
- xᵢ is the observed sensor value  
- x̂ᵢ is the predicted sensor value  
- n is the number of data points  

A high MAE indicates abnormal sensor behavior or a potential fault.

---

### 2. **LLM-Powered Diagnostic Engine (Gemini API)**
- Uses Google’s `generativelanguage.googleapis.com` endpoint (`gemini-2.5-flash-preview-05-20`).
- Prompt engineering integrates:
  - Anomaly score  
  - Top deviating sensors  
- Returns structured text containing:
  - **Expert Diagnosis**  
  - **Probable Root Cause**

Example Prompt:

---

### 3. **Streamlit User Interface**
- **Tab 1 – Test Data Analysis:**  
  Load and visualize engines from NASA FD001–FD004 datasets.  
- **Tab 2 – Live Data Analysis:**  
  Paste or stream real-time sensor readings for anomaly detection.  
- Interactive plots built with **Plotly** and **Matplotlib**.
- PDF export via FPDF library.

---

## 📦 Folder Structure

```plaintext
project_root/
│
├── app.py                     ← Streamlit app (main UI)
├── expert_rules.py
├── run_combined.py
├── run_project_fd001.py
├── run_project_fd002.py
├── run_project_fd003.py
├── run_project_fd004.py
│
├── data/
│   ├── readme.txt
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   ├── Damage Propagation Modeling.pdf
│   └── ... (same for FD002–FD004)
│
├── models/                    ← Trained models (.h5)
├── plots/                     ← Generated charts/reports
├── venv/                      ← Virtual environment
├── .env                       ← Contains secrets (DO NOT COMMIT)
└── .gitignore                 ← Controls what’s excluded from Git
```

---

## 🔒 Security and Privacy

**Sensitive components not committed:**
- `.env` → contains `GOOGLE_API_KEY`
- `/models/` → stores trained weights
- `/data/` → raw NASA datasets
- `/plots/` → runtime outputs
- `/venv/` → local environment

All secrets are loaded via:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
```
## 🧪 Example Workflow

1. **Select Dataset** → e.g., `FD002`  
2. **Choose Engine ID**  
3. **Click Generate PDF Report**

**The App Performs:**
- Computes reconstruction **MAE (Mean Absolute Error)**
- Identifies **abnormal sensors**
- Queries **Gemini API** for natural-language diagnosis
- Outputs a **formatted PDF report** with plots and LLM explanations

---

## 📊 Example Output

**Anomaly Score:** `0.1624`  
**Top Deviating Sensors:** `s3`, `s9`, `s14`

**LLM Output:**
> **Expert Diagnosis:** Compressor efficiency degradation detected.  
> **Probable Root Cause:** Stage 2 turbine wear leading to increased vibration and thermal imbalance.

**Generated Report:**  
`Anomaly_Report_Engine_FD002_123.pdf`

---

## 🧠 Dependencies

| Library | Purpose |
|----------|----------|
| `tensorflow.keras` | LSTM Autoencoder architecture |
| `sklearn.preprocessing` | MinMax scaling |
| `streamlit` | Interactive web application |
| `plotly`, `matplotlib` | Visualization |
| `fpdf` | PDF generation |
| `requests` | Gemini API calls |
| `python-dotenv` | Secure environment variable management |
| `numpy`, `pandas` | Data manipulation |

**Install via:**
```bash
pip install -r requirements.txt
```
## 🧠 Author

**Arya Keskar**  
📧 dcaryakeskar@gmail.com
