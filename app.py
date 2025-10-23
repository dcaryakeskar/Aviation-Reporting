import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import base64
import json
import requests
import plotly.express as px
from dotenv import load_dotenv


# --- Configuration ---
MASTER_MODEL_PATH = 'models/lstm_autoencoder_model_combined.h5'
SEQ_LENGTH = 50
SENSOR_COLS = [f's{i}' for i in range(1, 22)]
DATASET_IDS = ['FD001', 'FD002', 'FD003', 'FD004']

# --- PDF Report Generation Class ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AI-Powered Aviation Diagnostic Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, content):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 8, content) # Reduced line height for better formatting
        self.ln()

    def add_plot(self, path, width=150):
        self.image(path, x=None, y=None, w=width)
        self.ln()

# --- Helper Functions ---

@st.cache_data
def get_master_scaler():
    all_train_df = []
    col_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + SENSOR_COLS
    for ds_id in DATASET_IDS:
        train_path = f'data/train_{ds_id}.txt'
        df_train = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
        df_train['engine_id'] = df_train['engine_id'].apply(lambda x: f"{ds_id}_{x}")
        all_train_df.append(df_train)
    combined_df = pd.concat(all_train_df, ignore_index=True)
    scaler = MinMaxScaler()
    scaler.fit(combined_df[SENSOR_COLS])
    return scaler

@st.cache_data
def load_test_data(ds_id):
    test_path = f'data/test_{ds_id}.txt'
    col_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + SENSOR_COLS
    df_test = pd.read_csv(test_path, sep=r'\s+', header=None, names=col_names)
    return df_test

@st.cache_resource
def load_master_model():
    if not os.path.exists(MASTER_MODEL_PATH):
        return None
    model = load_model(MASTER_MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='mae')
    return model

# --- Gemini AI Investigation Function ---
def get_llm_diagnosis(anomaly_score, deviating_sensors):
    """Calls the Gemini LLM to get an expert diagnosis."""
    prompt = f"""
    You are an expert aviation maintenance engineer analyzing sensor data from a turbofan engine.
    An AI anomaly detection model has produced the following results:

    - Overall Anomaly Score (Reconstruction Error): {anomaly_score:.4f}
    - Top 3 Deviating Sensors: {', '.join(deviating_sensors)}

    Based on this information, provide a concise technical diagnosis. Identify the most likely faulty component and the probable root cause. Structure your response in two parts: "Expert Diagnosis" and "Probable Root Cause".
    """
    
    try:
        load_dotenv() 
        api_key = os.getenv("GOOGLE_API_KEY")
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
        
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {'Content-Type': 'application/json'}

        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        if "candidates" in response_json and response_json["candidates"]:
            content = response_json["candidates"][0]["content"]["parts"][0]["text"]
            return content.replace('**', '')
        else:
            return "AI diagnosis could not be generated. The API response was invalid."

    except requests.exceptions.RequestException as e:
        return f"An error occurred while contacting the AI investigator: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# --- Automated Investigation Function ---
def investigate_anomaly(original_sequence, predicted_sequence, sensor_names, anomaly_score):
    sensor_errors = np.mean(np.abs(original_sequence - predicted_sequence), axis=0)
    top_3_indices = np.argsort(sensor_errors)[-3:]
    deviating_sensors = [sensor_names[i] for i in reversed(top_3_indices)]
    
    diagnosis = get_llm_diagnosis(anomaly_score, deviating_sensors)

    summary = "The anomaly score is primarily driven by significant deviations in the following sensors:\n\n"
    for sensor in deviating_sensors:
        summary += f"- {sensor}\n"
        
    summary += f"\n**AI-Powered Investigation:**\n{diagnosis}"
    return summary, diagnosis

# Function to generate the PDF
def generate_pdf_report(engine_id, anomaly_score, sensor_plot_path, eval_plot_path, llm_analysis):
    pdf = PDFReport()
    pdf.add_page()
    
    pdf.chapter_title('1. Summary of Findings')
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_text = (
        f"Report Date: {report_date}\n"
        f"Engine ID: {engine_id}\n\n"
        "The AI model has flagged this engine with a high anomaly score. An automated investigation using a Large Language Model (LLM) has provided a detailed diagnosis below."
    )
    pdf.chapter_body(summary_text)

    pdf.chapter_title('2. Key Metric: Anomaly Score')
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f"{anomaly_score:.4f}", 0, 1, 'C')
    pdf.ln(5)

    pdf.chapter_title('3. AI-Powered Diagnostic Analysis')
    pdf.chapter_body(llm_analysis)

    pdf.chapter_title('4. Supporting Visual Evidence')
    
    if sensor_plot_path and os.path.exists(sensor_plot_path):
        pdf.add_plot(sensor_plot_path, width=160)
    
    if eval_plot_path and os.path.exists(eval_plot_path):
        pdf.add_plot(eval_plot_path, width=160)

    return bytes(pdf.output())


# --- Main Application ---
st.set_page_config(page_title="Aviation Anomaly Detector - Master Model", layout="wide")
st.title("✈️ Aviation Anomaly Detector (Master Model)")

master_model = load_master_model()
master_scaler = get_master_scaler()

if master_model is None:
    st.error(f"Master model file not found! Please run `run_project_combined.py` first.")
else:
    tab1, tab2 = st.tabs(["Analyze Test Data", "Analyze Live Data"])

    # --- TAB 1: Analyze Pre-loaded Test Data ---
    with tab1:
        st.header("Analyze Engines from NASA Test Datasets")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Controls")
            selected_dataset_id = st.selectbox("Select Dataset:", DATASET_IDS, key="tab1_dataset")
            df_test = load_test_data(selected_dataset_id)
            selected_engine_id = st.selectbox("Select Engine:", df_test['engine_id'].unique(), key="tab1_engine")
            
            st.subheader("Actions")
            generate_report_button = st.button("Generate PDF Report", key="tab1_report_btn")

        with col2:
            st.subheader(f"Inspecting Engine {selected_engine_id} from Dataset {selected_dataset_id}")
            engine_data = df_test[df_test['engine_id'] == selected_engine_id].copy()
            engine_data_scaled = master_scaler.transform(engine_data[SENSOR_COLS])
            
            if len(engine_data_scaled) >= SEQ_LENGTH:
                last_sequence = engine_data_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, len(SENSOR_COLS))
            else:
                padded_data = np.zeros((SEQ_LENGTH, len(SENSOR_COLS)))
                padded_data[-len(engine_data_scaled):] = engine_data_scaled
                last_sequence = padded_data.reshape(1, SEQ_LENGTH, len(SENSOR_COLS))
            
            prediction = master_model.predict(last_sequence, verbose=0)
            mae = np.mean(np.abs(prediction - last_sequence))
            
            st.metric(label="Anomaly Score", value=f"{mae:.4f}")

            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                selected_sensor = st.selectbox("Select Sensor:", SENSOR_COLS, key="tab1_sensor")
                fig = px.line(engine_data, x='cycle', y=selected_sensor, title=f"Sensor '{selected_sensor}' Readings")
                st.plotly_chart(fig, use_container_width=True)
            with viz_col2:
                eval_plot_path = f'plots/evaluation_plot_{selected_dataset_id}_master_model.png'
                if os.path.exists(eval_plot_path):
                    st.image(eval_plot_path, caption=f"Evaluation Context for {selected_dataset_id}")
            
            if generate_report_button:
                with st.spinner("Contacting AI investigator and generating report..."):
                    _, llm_diagnosis = investigate_anomaly(last_sequence[0], prediction[0], SENSOR_COLS, mae)
                    
                    sensor_plot_path_report = 'plots/temp_report_sensor_plot.png'
                    plt.figure(figsize=(8, 4))
                    plt.plot(engine_data['cycle'], engine_data[selected_sensor])
                    plt.title(f"Sensor '{selected_sensor}' Readings for Engine {selected_engine_id}")
                    plt.savefig(sensor_plot_path_report)
                    plt.close()
                    
                    pdf_data = generate_pdf_report(f"{selected_engine_id} ({selected_dataset_id})", mae, sensor_plot_path_report, eval_plot_path, llm_diagnosis)
                    
                    report_filename = f'Anomaly_Report_Engine_{selected_engine_id}_{selected_dataset_id}.pdf'
                    st.success("Report is ready!")
                    st.download_button(label="Download PDF Report", data=pdf_data, file_name=report_filename, mime="application/pdf")

    # --- TAB 2: Analyze New "Live" Data (UPGRADED) ---
    with tab2:
        st.header("Analyze New Sensor Data Sequence")
        st.info(f"Paste exactly {SEQ_LENGTH} lines of {len(SENSOR_COLS)} sensor readings, separated by spaces.")
        
        pasted_data = st.text_area("Paste sensor data here:", height=300, placeholder="Example:\n518.67 642.15 1591.82 ... (21 values total)\n518.67 642.35 1587.99 ...\n(and so on for 50 lines)")
        
        if st.button("Analyze Pasted Data", key="tab2_analyze_btn"):
            if pasted_data:
                try:
                    lines = pasted_data.strip().split('\n')
                    if len(lines) != SEQ_LENGTH:
                        st.error(f"Error: Please paste exactly {SEQ_LENGTH} lines of data. You pasted {len(lines)}.")
                    else:
                        data_array = np.loadtxt(lines)
                        if data_array.shape != (SEQ_LENGTH, len(SENSOR_COLS)):
                            st.error(f"Error: Each line must contain exactly {len(SENSOR_COLS)} sensor values. Check your data format.")
                        else:
                            with st.spinner("Analyzing new data..."):
                                data_scaled = master_scaler.transform(data_array)
                                sequence = data_scaled.reshape(1, SEQ_LENGTH, len(SENSOR_COLS))
                                prediction = master_model.predict(sequence, verbose=0)
                                mae = np.mean(np.abs(prediction - sequence))
                                
                                # Store results in session state to use them later
                                st.session_state['live_mae'] = mae
                                st.session_state['live_sequence'] = sequence
                                st.session_state['live_prediction'] = prediction
                                st.session_state['live_data_array'] = data_array
                                st.rerun() # Rerun to show the results below

                except Exception as e:
                    st.error(f"An error occurred while parsing the data: {e}. Please ensure it is formatted correctly.")
            else:
                st.warning("Please paste some data into the text area first.")

        # Display results after analysis
        if 'live_mae' in st.session_state:
            st.metric(label="Anomaly Score for New Data", value=f"{st.session_state['live_mae']:.4f}")
            
            # --- NEW: Add graph for live data ---
            st.subheader("Live Sensor Data Visualization")
            live_df = pd.DataFrame(st.session_state['live_data_array'], columns=SENSOR_COLS)
            live_df['cycle'] = range(1, len(live_df) + 1)
            selected_sensor_live = st.selectbox("Select Sensor to Plot:", SENSOR_COLS, key="tab2_sensor")
            fig_live = px.line(live_df, x='cycle', y=selected_sensor_live, title=f"Live Data for Sensor '{selected_sensor_live}'")
            st.plotly_chart(fig_live, use_container_width=True)

            # Get LLM Diagnosis
            _, llm_diagnosis = investigate_anomaly(st.session_state['live_sequence'][0], st.session_state['live_prediction'][0], SENSOR_COLS, st.session_state['live_mae'])
            st.subheader("AI-Powered Investigation Results")
            st.markdown(llm_diagnosis)

            # --- NEW: Upgraded PDF Report for live data ---
            st.subheader("Generate Report for New Data")
            if st.button("Generate PDF", key="tab2_report_btn"):
                 with st.spinner("Generating PDF..."):
                    # Create a temporary plot for the report
                    live_plot_path = 'plots/temp_live_report_plot.png'
                    plt.figure(figsize=(8, 4))
                    plt.plot(live_df['cycle'], live_df[selected_sensor_live])
                    plt.title(f"Live Data Sensor '{selected_sensor_live}' Readings")
                    plt.savefig(live_plot_path)
                    plt.close()

                    pdf_data = generate_pdf_report("Live Data Input", st.session_state['live_mae'], live_plot_path, None, llm_diagnosis)
                    st.success("Report for live data is ready!")
                    st.download_button(label="Download Live Data Report", data=pdf_data, file_name="Live_Data_Report.pdf", mime="application/pdf")

