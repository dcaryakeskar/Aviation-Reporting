from collections import Counter
import pandas as pd
import spacy

# Load the small English NLP model
# Make sure to run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model... (this will only happen once)")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- Expert Domain Knowledge ---

# 1. Expanded mapping of sensors to their primary associated components
SENSOR_COMPONENT_MAP = {
    'Fan': ['s11'], # s11: Physical fan speed
    'LPC': ['s2', 's12'], # s2: LPC Outlet Temp, s12: Physical core speed
    'HPC': ['s3', 's4', 's9', 's15'], # s3/s4/s9/s15: HPC Temps and Pressures
    'Combustor/Fuel': ['s7', 's17'], # s7/s17: Ratios of fuel flow to pressure
    'HPT': ['s14', 's20'], # s14: HPT outlet temp, s20: HPT coolant bleed
    'LPT': ['s8', 's21'] # s8: LPT outlet temp, s21: LPT coolant bleed
}

# 2. Keywords for NLP search, linked to diagnoses
DIAGNOSIS_KEYWORDS = {
    "HPC Stall": ["hpc", "stall", "surge", "pressure", "vibration"],
    "HPC Degradation": ["hpc", "degradation", "wear", "pressure", "efficiency"],
    "HPT Degradation": ["hpt", "turbine", "temp", "coolant", "blade"],
    "LPT Degradation": ["lpt", "turbine", "temp", "coolant", "bleed"],
    "Fan issue": ["fan", "lpc", "vibration", "imbalance", "speed"],
    "Fuel System Anomaly": ["fuel", "combustor", "nozzle", "flow"]
}

# 3. Diagnostic Logic
def diagnose_fault(deviating_sensors):
    """
    Analyzes a list of the top deviating sensors and returns a detailed technical diagnosis.
    """
    sensor_set = set(deviating_sensors)

    # --- Expert Rules based on specific sensor combinations ---
    if {'s4', 's9', 's14'}.issubset(sensor_set) or {'s4', 's9', 's15'}.issubset(sensor_set):
        return "HPC Stall"
    if {'s11', 's15', 's4'}.issubset(sensor_set):
        return "HPC Degradation"
    if {'s14', 's20'}.issubset(sensor_set):
        return "HPT Degradation"
    if {'s8', 's21'}.issubset(sensor_set):
        return "LPT Degradation"
    if {'s11', 's12'}.issubset(sensor_set) or {'s2', 's8'}.issubset(sensor_set):
        return "Fan issue"
    if {'s7', 's17'}.issubset(sensor_set):
        return "Fuel System Anomaly"

    # --- Fallback to component-level analysis ---
    possible_components = [comp for sensor in deviating_sensors for comp, sensors in SENSOR_COMPONENT_MAP.items() if sensor in sensors]
    if possible_components:
        most_likely = Counter(possible_components).most_common(1)[0][0]
        return f"General {most_likely} Anomaly"
        
    return "Undetermined Anomaly"

# 4. NLP function to find corroborating evidence
def find_corroborating_evidence(dataset_id, engine_id, diagnosis):
    """Searches maintenance logs for text evidence related to a diagnosis."""
    try:
        logs_df = pd.read_csv('data/maintenance_logs.csv')
    except FileNotFoundError:
        return "Maintenance log file not found."

    engine_logs = logs_df[(logs_df['dataset_id'] == dataset_id) & (logs_df['engine_id'] == engine_id)]
    
    if engine_logs.empty:
        return "No maintenance logs found for this engine."

    keywords = []
    for diag_key, kw_list in DIAGNOSIS_KEYWORDS.items():
        if diag_key in diagnosis:
            keywords.extend(kw_list)
    
    if not keywords:
        return "No specific keywords to search for this diagnosis."

    for index, row in engine_logs.iterrows():
        log_text = row['log_entry'].lower()
        if any(keyword in log_text for keyword in keywords):
            return f"Found relevant log entry from {row['timestamp']}:\n'{row['log_entry']}'"
            
    return "No corroborating text evidence found in maintenance logs."
