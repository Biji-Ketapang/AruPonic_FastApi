import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from datetime import datetime, timedelta
from pydantic import BaseModel
from dotenv import load_dotenv

# --- 1. KONFIGURASI DATABASE & AI ---

# Konfigurasi Database Supabase
DB_USER = "postgres.ehnfcusiaeshdqeywkjp"
DB_PASS = "Aruponic"
DB_HOST = "aws-1-ap-southeast-1.pooler.supabase.com"
DB_PORT = "6543"
DB_NAME = "postgres"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Gunakan binary_parameters=no untuk kompatibilitas pooler mode transaction
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Definisi Custom Layer (Wajib ada)
class AttentionBlock(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(AttentionBlock, self).build(input_shape)

    def call(self, x):
        e = tf.math.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape

# Load Model & Scaler
try:
    print("Loading AI Model & Scaler...")
    # Pastikan file ada di folder yang sama
    model = load_model('aruponic_mscnn_gru_model.keras', custom_objects={'AttentionBlock': AttentionBlock})
    scaler = joblib.load('aruponic_scaler.pkl')
    print("AI Resources Loaded Successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    model = None
    scaler = None

# Parameter Model
INPUT_WIDTH = 24  # 2 Jam input
FORECAST_STEPS = 72 # 6 Jam output
TARGET_COLUMNS = ['pH', 'ec', 'orp', 'turbidity', 'temperature', 'do']

# --- 2. SETUP FASTAPI ---

app = FastAPI(title="Aruponic Multi-Horizon AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. HELPER FUNCTIONS ---

def fetch_recent_data(limit=50):
    query = text('SELECT * FROM "Aruponic" ORDER BY timestamp DESC LIMIT :limit')
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"limit": limit})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        # Preprocessing & Fix Warning "UserWarning"
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=False)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df.set_index('timestamp', inplace=True)
        
        # Fix FutureWarning "T" -> "min"
        df_resampled = df.resample('5min').mean().interpolate(method='linear')
        
        return df_resampled
    except Exception as e:
        print(f"DB Error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def prepare_features(df):
    if len(df) < INPUT_WIDTH:
        raise HTTPException(status_code=400, detail="Data kurang untuk prediksi.")
    
    df_input = df.tail(INPUT_WIDTH).copy()
    
    sensor_data = df_input[TARGET_COLUMNS].values
    sensor_scaled = scaler.transform(sensor_data)
    
    day = 24 * 60 * 60
    timestamp_s = df_input.index.map(pd.Timestamp.timestamp)
    day_sin = np.sin(timestamp_s * (2 * np.pi / day)).values.reshape(-1, 1)
    day_cos = np.cos(timestamp_s * (2 * np.pi / day)).values.reshape(-1, 1)
    
    final_input = np.hstack([sensor_scaled, day_sin, day_cos])
    return np.expand_dims(final_input, axis=0), df_input.index[-1]

def get_trend_status(current, future):
    """Menentukan status tren (Naik/Turun/Stabil)"""
    # Ubah ke float python agar aman
    curr = float(current)
    fut = float(future)
    diff = fut - curr
    percent = (diff / curr) * 100 if curr != 0 else 0
    
    if percent > 5.0: return "Naik Tajam"
    if percent > 1.0: return "Naik"
    if percent < -5.0: return "Turun Tajam"
    if percent < -1.0: return "Turun"
    return "Stabil"

# --- 4. API ENDPOINTS ---

@app.get("/")
def root():
    return {"status": "Aruponic AI Ready", "horizons": ["1h", "3h", "6h"]}

@app.get("/history")
def get_history():
    """Endpoint untuk grafik data historis (Misal 6 jam terakhir)"""
    df = fetch_recent_data(limit=100) # Ambil agak banyak
    df_tail = df.tail(72) # Ambil 6 jam terakhir
    
    result = []
    for ts, row in df_tail.iterrows():
        result.append({
            "timestamp": ts.strftime('%Y-%m-%d %H:%M'),
            **row.to_dict()
        })
    return {"data": result}

@app.get("/predict")
def predict_multi_horizon():
    """
    Output khusus untuk 3 Horizon waktu:
    - 1 Jam (Step 12)
    - 3 Jam (Step 36)
    - 6 Jam (Step 72)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model AI belum siap.")

    # 1. Ambil Data
    df = fetch_recent_data(limit=40)
    current_vals = df.iloc[-1] # Nilai sensor terakhir (real)
    
    # 2. Prediksi
    X_input, last_ts = prepare_features(df)
    prediction_scaled = model.predict(X_input, verbose=0)
    prediction_real = scaler.inverse_transform(prediction_scaled[0])
    
    # 3. Ekstrak Multi-Horizon Indices
    # Data per 5 menit.
    # 1 Jam = 60 / 5 = 12 steps (index 11)
    # 3 Jam = 180 / 5 = 36 steps (index 35)
    # 6 Jam = 360 / 5 = 72 steps (index 71)
    
    indices = {
        "1_hour": 11,
        "3_hours": 35,
        "6_hours": 71
    }
    
    result = {
        "meta": {
            "last_sensor_time": last_ts.strftime('%Y-%m-%d %H:%M'),
            "model": "Residual MS-CNN-Att-GRU"
        },
        "current_conditions": {},
        "predictions": {}
    }

    # Isi Kondisi Saat Ini (dengan cast float agar JSON valid)
    for col in TARGET_COLUMNS:
        result["current_conditions"][col] = round(float(current_vals[col]), 3)

    # Isi Prediksi Horizon
    for horizon_name, idx in indices.items():
        horizon_data = {}
        forecast_ts = last_ts + timedelta(minutes=(idx + 1) * 5)
        
        for i, col in enumerate(TARGET_COLUMNS):
            # FIX UTAMA: Menggunakan float() untuk konversi numpy.float32
            future_val = float(prediction_real[idx][i])
            current_val = float(current_vals[col])
            
            horizon_data[col] = {
                "value": round(future_val, 3),
                "trend": get_trend_status(current_val, future_val),
                "diff": round(future_val - current_val, 3)
            }
        
        result["predictions"][horizon_name] = {
            "timestamp": forecast_ts.strftime('%Y-%m-%d %H:%M'),
            "data": horizon_data
        }
        
    return result
# --- PANDUAN ---
# Output JSON akan terstruktur seperti ini:
# {
#   "current": { "pH": 6.8, ... },
#   "predictions": {
#      "1_hour": { "timestamp": "...", "data": { "pH": { "value": 6.7, "trend": "Stabil" }, ... } },
#      "3_hours": { ... },
#      "6_hours": { ... }
#   }
# }