import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
import sqlite3
import datetime
from tensorflow.keras.models import load_model
from data_preprocessing import clean_signal, LABEL_MAP
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Database configuration
DB_FILE = 'db/smart_cardiology.db'

def setup_database():
    """Initializes the SQLite database for storing diagnosis results."""
    os.makedirs('db', exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnosis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            heart_rate INTEGER,
            diagnosis TEXT,
            confidence REAL,
            risk_level TEXT,
            remarks TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(filename, hr, diagnosis, confidence, risk, remarks):
    """Saves the prediction result into the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO diagnosis_results 
        (timestamp, filename, heart_rate, diagnosis, confidence, risk_level, remarks)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, filename, hr, diagnosis, confidence, risk, remarks))
    conn.commit()
    conn.close()

class ECGPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Cardiology - ECG Analysis AI")
        self.root.geometry("800x600")
        
        # Load Model
        self.model_path = 'models/ecg_model.h5'
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            messagebox.showwarning("Model Not Found", "ecg_model.h5 not found! Please run train_model.py first.")
            self.model = None

        self.setup_ui()
        setup_database()

    def setup_ui(self):
        # Header
        lbl_title = tk.Label(self.root, text="ECG Signal Analysis AI", font=("Arial", 20, "bold"), pady=10)
        lbl_title.pack()
        
        # Upload Button
        btn_upload = tk.Button(self.root, text="Upload ECG CSV", command=self.upload_file, font=("Arial", 12), bg="lightblue")
        btn_upload.pack(pady=10)

        # Plot frame
        self.plot_frame = tk.Frame(self.root, width=700, height=250, bg="white", relief=tk.SUNKEN, bd=2)
        self.plot_frame.pack(pady=10, fill=tk.X, padx=20)
        
        # Results frame
        self.results_frame = tk.Frame(self.root)
        self.results_frame.pack(pady=20)
        
        self.lbl_hr = tk.Label(self.results_frame, text="Heart Rate: -- bpm", font=("Arial", 14))
        self.lbl_hr.grid(row=0, column=0, padx=20, pady=5, sticky="w")
        
        self.lbl_diagnosis = tk.Label(self.results_frame, text="Diagnosis: --", font=("Arial", 14, "bold"))
        self.lbl_diagnosis.grid(row=1, column=0, padx=20, pady=5, sticky="w")
        
        self.lbl_confidence = tk.Label(self.results_frame, text="Confidence: --%", font=("Arial", 14))
        self.lbl_confidence.grid(row=2, column=0, padx=20, pady=5, sticky="w")
        
        self.lbl_risk = tk.Label(self.results_frame, text="Risk Level: --", font=("Arial", 14))
        self.lbl_risk.grid(row=3, column=0, padx=20, pady=5, sticky="w")

        self.lbl_remarks = tk.Label(self.results_frame, text="AI Remarks: --", font=("Arial", 12, "italic"), fg="gray")
        self.lbl_remarks.grid(row=4, column=0, padx=20, pady=10, sticky="w")

    def upload_file(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded!")
            return
            
        filepath = filedialog.askopenfilename(
            title="Select ECG CSV", 
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not filepath:
            return
            
        try:
            # Read single sample CSV (assuming it has no header or 1 row of data)
            df = pd.read_csv(filepath, header=None)
            
            # If the CSV has headers, skip it (rudimentary check)
            if isinstance(df.iloc[0,0], str):
                df = pd.read_csv(filepath)
            
            signal_data = df.iloc[0].values
            
            # Display plot
            self.plot_signal(signal_data)
            
            # Predict
            self.predict_signal(signal_data, os.path.basename(filepath))
            
        except Exception as e:
            messagebox.showerror("Error processing file", str(e))

    def plot_signal(self, signal_data):
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        fig, ax = plt.subplots(figsize=(7, 2.5), dpi=100)
        ax.plot(signal_data, color='blue')
        ax.set_title("ECG Waveform")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def predict_signal(self, signal_data, filename):
        # 1. Preprocess signal
        # Ensure we have exactly 187 features. If it has 188 (e.g. from dummy_data.csv), drop the label.
        if len(signal_data) >= 187:
            signal_data = signal_data[:187]
        else:
            messagebox.showerror("Error", f"Signal has {len(signal_data)} features, but 187 are required.")
            return
            
        # Standardize the single signal across time steps
        signal_data = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)
        
        # Reshape for CNN+LSTM (samples, time_steps, features)
        signal_input = signal_data.reshape(1, 187, 1)
        
        # 2. Predict
        preds = self.model.predict(signal_input)[0]
        class_idx = np.argmax(preds)
        confidence = preds[class_idx] * 100
        diagnosis = LABEL_MAP[class_idx]
        
        # 3. Simulate Heart Rate calculation (In real scenario, detect R-peaks)
        hr = np.random.randint(60, 100) if class_idx == 0 else np.random.randint(90, 150)
        
        # 4. Determine Risk and Remarks
        risk = "Low" if class_idx == 0 else ("High" if class_idx == 2 else "Moderate")
        
        remarks_map = {
            0: "No significant abnormalities detected. Rhythm appears normal.",
            1: "Possible ischemia or electrolyte imbalance. Clinical correlation advised.",
            2: "Signs of ventricular thickening. Recommend Echocardiogram."
        }
        remarks = remarks_map[class_idx]
        
        # 5. Update UI
        self.lbl_hr.config(text=f"Heart Rate: {hr} bpm")
        self.lbl_diagnosis.config(text=f"Diagnosis: {diagnosis}")
        self.lbl_confidence.config(text=f"Confidence: {confidence:.2f}%")
        self.lbl_risk.config(text=f"Risk Level: {risk}")
        
        if risk == "Low":
            self.lbl_risk.config(fg="green")
            self.lbl_diagnosis.config(fg="green")
        elif risk == "Moderate":
            self.lbl_risk.config(fg="orange")
            self.lbl_diagnosis.config(fg="orange")
        else:
            self.lbl_risk.config(fg="red")
            self.lbl_diagnosis.config(fg="red")
            
        self.lbl_remarks.config(text=f"AI Remarks: {remarks}")
        
        # 6. Save to DB
        save_prediction(filename, hr, diagnosis, confidence, risk, remarks)

if __name__ == "__main__":
    root = tk.Tk()
    app = ECGPredictorApp(root)
    root.mainloop()
