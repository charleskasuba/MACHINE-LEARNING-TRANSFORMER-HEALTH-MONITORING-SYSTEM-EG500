"""
TRANSFORMER HEALTH MONITORING SYSTEM
CNN INTEGRATED VERSION - FULLY CORRECTED
"""

import time
import json
import numpy as np
from datetime import datetime, timedelta
import threading
import random
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from collections import deque
import warnings
import joblib
import os
import requests
import sqlite3
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ============================================
# DEEPSEEK API CONFIGURATION
# ============================================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# ============================================
# BIRD EMAIL ALERTS CONFIGURATION
# ============================================

BIRD_API_KEY = os.environ.get("BIRD_API_KEY", "bk_eu1_1rjPniF2y4qEF2JjyKIPCcW3eom4Q")
BIRD_API_URL = "https://eu1.platform.bird.com/v1/email/messages"
BIRD_FROM_EMAIL = os.environ.get("BIRD_FROM_EMAIL", "onboarding@messagebird.dev")
ALERT_EMAIL_TO = os.environ.get("ALERT_EMAIL_TO", "charleskasuba81@gmail.com")
last_email_time = {}

def send_email_alert(subject, body_html):
    """Send alert email via Bird API with rate limiting"""
    now = time.time()
    key = subject[:30]
    if key in last_email_time and now - last_email_time[key] < 300:
        return
    try:
        resp = requests.post(
            BIRD_API_URL,
            headers={
                "Authorization": f"Bearer {BIRD_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": BIRD_FROM_EMAIL,
                "to": [ALERT_EMAIL_TO],
                "subject": subject,
                "html": body_html,
            },
            timeout=10,
        )
        last_email_time[key] = now
        print(f"📧 Email alert sent ({resp.status_code}): {subject}")
    except Exception as e:
        print(f"📧 Email alert failed: {e}")

# ============================================
# SQLITE DATABASE
# ============================================

DB_PATH = os.environ.get("DB_PATH", "data.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            primary_current REAL,
            primary_voltage REAL,
            primary_power_kw REAL,
            primary_pf REAL,
            secondary_current REAL,
            secondary_voltage REAL,
            secondary_power_kw REAL,
            secondary_pf REAL,
            temperature REAL,
            humidity REAL,
            efficiency REAL,
            fault_overcurrent INTEGER,
            fault_overtemp INTEGER,
            flame INTEGER,
            health_status TEXT,
            health_confidence REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            type TEXT,
            message TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_reading(data):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO readings (
                timestamp, primary_current, primary_voltage, primary_power_kw, primary_pf,
                secondary_current, secondary_voltage, secondary_power_kw, secondary_pf,
                temperature, humidity, efficiency,
                fault_overcurrent, fault_overtemp, flame,
                health_status, health_confidence
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            data.get('primary_current', 0),
            data.get('primary_voltage', 230),
            data.get('primary_power_kw', 0),
            data.get('primary_pf', 0.85),
            data.get('secondary_current', 0),
            data.get('secondary_voltage', 19),
            data.get('secondary_power_kw', 0),
            data.get('secondary_pf', 0.82),
            data.get('temperature', 27),
            data.get('humidity', 45),
            data.get('efficiency', 85),
            1 if data.get('fault_overcurrent', False) else 0,
            1 if data.get('fault_overtemp', False) else 0,
            1 if data.get('flame', False) else 0,
            data.get('health_status', 'Healthy'),
            data.get('health_confidence', 85)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB save error: {e}")

def save_alert(alert):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO alerts (timestamp, type, message) VALUES (?,?,?)",
                     (alert.get('timestamp', ''), alert.get('type', ''), alert.get('message', '')))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB alert save error: {e}")

def cleanup_old_records():
    """Delete readings and alerts older than 30 days"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cutoff = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        r = conn.execute("DELETE FROM readings WHERE timestamp < ?", (cutoff,)).rowcount
        a = conn.execute("DELETE FROM alerts WHERE timestamp < ?", (cutoff,)).rowcount
        conn.commit()
        conn.close()
        if r or a:
            print(f"🧹 Cleaned up {r} readings, {a} alerts older than 30 days")
    except Exception as e:
        print(f"Cleanup error: {e}")

def cleanup_loop():
    """Run cleanup every 6 hours"""
    while True:
        time.sleep(21600)
        cleanup_old_records()

def get_readings_from_db(limit=100):
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT * FROM readings ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        cols = ['id','timestamp','primary_current','primary_voltage','primary_power_kw','primary_pf',
                'secondary_current','secondary_voltage','secondary_power_kw','secondary_pf',
                'temperature','humidity','efficiency','fault_overcurrent','fault_overtemp',
                'flame','health_status','health_confidence']
        return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        print(f"DB read error: {e}")
        return []

# ============================================
# LOAD TRAINED CNN MODEL
# ============================================

class CNNHealthPredictor:
    """Wrapper for trained CNN model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.is_loaded = False
        
        # Try to load the trained model files
        model_paths = [
            'transformer_health_monitor_model_updated.h5',
            'best_transformer_model_updated.h5',
            'transformer_health_monitor_model.h5'
        ]
        
        scaler_paths = [
            'scaler_updated.pkl',
            'scaler.pkl'
        ]
        
        label_encoder_paths = [
            'label_encoder_updated.pkl',
            'label_encoder.pkl'
        ]
        
        feature_cols_paths = [
            'feature_columns_updated.pkl',
            'feature_columns.pkl'
        ]
        
        # Load model
        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.model = load_model(path)
                    print(f"✓ CNN Model loaded from: {path}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {path}: {e}")
        
        # Load scaler
        for path in scaler_paths:
            if os.path.exists(path):
                try:
                    self.scaler = joblib.load(path)
                    print(f"✓ Scaler loaded from: {path}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {path}: {e}")
        
        # Load label encoder
        for path in label_encoder_paths:
            if os.path.exists(path):
                try:
                    self.label_encoder = joblib.load(path)
                    print(f"✓ Label encoder loaded from: {path}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {path}: {e}")
        
        # Load feature columns
        for path in feature_cols_paths:
            if os.path.exists(path):
                try:
                    self.feature_columns = joblib.load(path)
                    print(f"✓ Feature columns loaded from: {path}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {path}: {e}")
        
        # Check if all components loaded
        if self.model and self.scaler and self.label_encoder:
            self.is_loaded = True
            print("✅ CNN Model fully loaded and ready for predictions!")
            print(f"   Model input shape: {self.model.input_shape}")
            print(f"   Number of classes: {len(self.label_encoder.classes_)}")
            print(f"   Classes: {list(self.label_encoder.classes_)}")
        else:
            print("⚠️ CNN Model not fully loaded - falling back to rule-based system")
    
    def preprocess_input(self, raw_data):
        """Convert raw sensor data to model input format"""
        if not self.is_loaded:
            return None
        
        try:
            # Calculate derived features
            primary_current = raw_data.get('primary_current', 0)
            secondary_current = raw_data.get('secondary_current', 0)
            primary_voltage = raw_data.get('primary_voltage', 230)
            secondary_voltage = raw_data.get('secondary_voltage', 19)
            primary_power = raw_data.get('primary_power_kw', 0)
            secondary_power = raw_data.get('secondary_power_kw', 0)
            temperature = raw_data.get('temperature', 27)
            humidity = raw_data.get('humidity', 45)
            primary_pf = raw_data.get('primary_pf', 0.85)
            secondary_pf = raw_data.get('secondary_pf', 0.82)
            
            # Engineered features (matching training)
            voltage_ratio = primary_voltage / max(secondary_voltage, 0.1)
            current_ratio = secondary_current / max(primary_current, 0.001)
            power_ratio = secondary_power / max(primary_power, 0.001)
            temp_humidity_index = temperature * (humidity / 100)
            power_factor_diff = abs(primary_pf - secondary_pf)
            apparent_power_ratio = (secondary_voltage * secondary_current) / \
                                    max(primary_voltage * primary_current, 0.001)
            
            # Create feature vector (16 features)
            features = np.array([[
                primary_voltage,
                primary_current,
                primary_power,
                primary_pf,
                secondary_voltage,
                secondary_current,
                secondary_power,
                secondary_pf,
                temperature,
                humidity,
                voltage_ratio,
                current_ratio,
                power_ratio,
                temp_humidity_index,
                power_factor_diff,
                apparent_power_ratio
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Reshape for CNN: (batch, features, channels)
            return features_scaled.reshape(1, 16, 1)
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def predict(self, raw_data):
        """Make prediction with confidence"""
        if not self.is_loaded:
            return None
        
        try:
            X = self.preprocess_input(raw_data)
            if X is None:
                return None
            
            # Get probabilities from CNN
            probabilities = self.model.predict(X, verbose=0)[0]
            
            # Get predicted class
            predicted_idx = np.argmax(probabilities)
            predicted_status = self.label_encoder.inverse_transform([predicted_idx])[0]
            confidence = float(probabilities[predicted_idx] * 100)
            
            # Get all probabilities
            all_probs = {}
            for status, prob in zip(self.label_encoder.classes_, probabilities):
                all_probs[status] = float(prob * 100)
            
            return {
                'predicted_status': predicted_status,
                'confidence': confidence,
                'probabilities': all_probs
            }
            
        except Exception as e:
            print(f"CNN Prediction error: {e}")
            return None

# Initialize CNN predictor
cnn_predictor = CNNHealthPredictor()

# ============================================
# DEEPSEEK CHATBOT FUNCTION
# ============================================

def get_deepseek_response(user_message, current_data, life_data, predictions):
    """Get response from DeepSeek API for chatbot"""
    
    # Prepare context about transformer
    context = f"""
You are an AI assistant for a Transformer Health Monitoring System. 
Be helpful, concise, and accurate. Use emojis where appropriate.

CURRENT TRANSFORMER STATUS:
- Health Status: {current_data.get('health_status', 'Unknown')}
- Health Confidence: {current_data.get('health_confidence', 0):.1f}%
- Temperature: {current_data.get('temperature', 0):.1f}°C
- Primary Current: {current_data.get('primary_current', 0):.1f}A
- Humidity: {current_data.get('humidity', 0):.1f}%
- Efficiency: {current_data.get('efficiency', 0):.1f}%
- Flame Sensor: {'🔥 ACTIVE - FIRE DETECTED!' if current_data.get('flame', False) else 'Normal'}
- Faults: {'Overcurrent' if current_data.get('fault_overcurrent', False) else ''} {'Overtemp' if current_data.get('fault_overtemp', False) else ''}

LIFE EXPECTANCY:
- Remaining: {life_data.get('remaining_years', 0)} years
- Health Score: {life_data.get('health_score', 0)}%
- Degradation Rate: {life_data.get('degradation_rate', 0)}%

FUTURE PREDICTIONS:
- 1 Month: {predictions[0]['projected_health_score'] if len(predictions) > 0 else 0}% ({predictions[0]['projected_status'] if len(predictions) > 0 else 'Unknown'})
- 3 Months: {predictions[1]['projected_health_score'] if len(predictions) > 1 else 0}% ({predictions[1]['projected_status'] if len(predictions) > 1 else 'Unknown'})
- 6 Months: {predictions[2]['projected_health_score'] if len(predictions) > 2 else 0}% ({predictions[2]['projected_status'] if len(predictions) > 2 else 'Unknown'})
- 12 Months: {predictions[3]['projected_health_score'] if len(predictions) > 3 else 0}% ({predictions[3]['projected_status'] if len(predictions) > 3 else 'Unknown'})

Temperature operating range: 0°C to 180°C
- Normal: 0-85°C (Optimal)
- Elevated: 85-105°C (Monitor)
- High: 105-130°C (Warning)
- Critical: 130-180°C (Emergency)

If FIRE is detected, emphasize emergency actions.
If humidity > 80%, warn about insulation risks.
Answer the user's question based on this data.
"""
    
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return fallback_response(user_message, current_data, life_data, predictions)
            
    except Exception as e:
        print(f"DeepSeek API error: {e}")
        return fallback_response(user_message, current_data, life_data, predictions)

def fallback_response(user_message, current_data, life_data, predictions):
    """Fallback response if API fails"""
    msg = user_message.lower()
    
    if current_data.get('flame', False):
        return "🚨 **EMERGENCY: FIRE DETECTED!** 🚨\n\nImmediate actions required:\n1. Shut down transformer immediately\n2. Call emergency services\n3. Evacuate the area\n4. Activate fire suppression\n\nDo NOT use water on electrical fire!"
    
    if 'health' in msg or 'status' in msg:
        return f"📊 Current health status: **{current_data.get('health_status', 'Unknown')}** with {current_data.get('health_confidence', 0):.1f}% confidence.\n\n{get_status_recommendation(current_data.get('health_status', 'Unknown'))}"
    
    if 'life' in msg or 'expectancy' in msg:
        return f"⏰ **Life Expectancy:** {life_data.get('remaining_years', 0)} years remaining\n• Health Score: {life_data.get('health_score', 0)}%\n• Degradation Rate: {life_data.get('degradation_rate', 0)}%\n• Est. Failure: {life_data.get('estimated_failure_date', 'Unknown')}"
    
    if 'temp' in msg:
        temp = current_data.get('temperature', 0)
        return f"🌡️ Temperature: {temp:.1f}°C (Range: 0-180°C)\n\n{get_temp_recommendation(temp)}"
    
    if 'humidity' in msg:
        hum = current_data.get('humidity', 0)
        return f"💧 Humidity: {hum:.1f}%\n\n{get_humidity_recommendation(hum)}"
    
    return "I can help you with transformer health info. Ask about:\n• Health status\n• Life expectancy\n• Temperature\n• Humidity\n• Predictions\n• Recommendations"

def get_status_recommendation(status):
    if status == 'Critical':
        return "🚨 **CRITICAL - Immediate action required!** Prepare for emergency shutdown."
    elif status == 'Warning':
        return "⚠️ **Warning** - Schedule maintenance within 24 hours."
    elif status == 'Monitor':
        return "📊 **Monitor** - Increase monitoring frequency."
    else:
        return "✅ **Healthy** - Continue regular monitoring."

def get_temp_recommendation(temp):
    if temp > 180:
        return "💀 CRITICAL! Temperature exceeds ABSOLUTE MAXIMUM (180°C)! IMMEDIATE SHUTDOWN REQUIRED!"
    elif temp > 130:
        return "🔥 CRITICAL! Temperature above 130°C. Emergency cooling required! Reduce load immediately."
    elif temp > 105:
        return "⚠️ HIGH TEMPERATURE: Above 105°C. Check cooling fans, reduce load."
    elif temp > 85:
        return "📈 Elevated temperature: Above 85°C. Monitor closely."
    else:
        return "✅ Temperature normal: Within optimal range (0-85°C)."

def get_humidity_recommendation(hum):
    if hum > 85:
        return "🚨 CRITICAL humidity! Risk of flashover! Install dehumidifier immediately."
    elif hum > 75:
        return "⚠️ High humidity - Monitor insulation resistance."
    elif hum > 65:
        return "📊 Elevated humidity - Consider ventilation."
    else:
        return "✅ Humidity normal."

# ============================================
# HELPER FUNCTIONS FOR LIFE EXPECTANCY
# ============================================

def calculate_life_expectancy(data):
    """Calculate life expectancy based on CNN prediction or rule-based"""
    
    flame = data.get('flame', False)
    health_status = data.get('health_status', 'Healthy')
    
    # EMERGENCY OVERRIDE - FIRE DETECTED
    if flame:
        return {
            'remaining_years': 0.0,
            'remaining_months': 0.0,
            'remaining_days': 0.0,
            'remaining_hours': 1.0,
            'aging_factor': 999.0,
            'degradation_rate': 100.0,
            'estimated_failure_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'confidence': 100.0,
            'health_score': 0.0
        }
    
    # Use CNN prediction if available
    if 'cnn_prediction' in data and data['cnn_prediction']:
        cnn_conf = data['cnn_prediction']['confidence']
        if health_status == 'Critical':
            health_score = min(cnn_conf, 30.0)
        elif health_status == 'Warning':
            health_score = min(cnn_conf, 60.0)
        elif health_status == 'Monitor':
            health_score = min(cnn_conf, 80.0)
        else:
            health_score = cnn_conf
    else:
        health_score = data.get('health_confidence', 85)
    
    temp = data.get('temperature', 27)
    current = data.get('primary_current', 0)
    humidity = data.get('humidity', 45)
    
    # Degradation calculation - temperature range 0-180°C
    if temp > 130:
        temp_factor = min(5.0, (temp - 85) / 30)
    elif temp > 85:
        temp_factor = (temp - 85) / 30
    else:
        temp_factor = 0.5
    
    current_factor = max(1, (current - 35) / 20) if current > 35 else 1
    humidity_factor = max(1, (humidity - 70) / 20) if humidity > 70 else 1
    
    degradation_rate = (temp_factor * current_factor * humidity_factor * 15) / 8760
    
    remaining_years = max(0.0, (health_score / 100) * 20 / (degradation_rate + 0.5))
    
    failure_date = datetime.now() + timedelta(days=remaining_years * 365)
    
    return {
        'remaining_years': round(remaining_years, 1),
        'remaining_months': round(remaining_years * 12, 1),
        'remaining_days': round(remaining_years * 365, 0),
        'aging_factor': round(1 + degradation_rate, 2),
        'degradation_rate': round(degradation_rate * 100, 1),
        'estimated_failure_date': failure_date.strftime('%Y-%m-%d %H:%M:%S'),
        'confidence': round(95 - degradation_rate * 20, 1),
        'health_score': round(health_score, 1)
    }

def generate_predictions(data):
    """Generate future predictions"""
    
    flame = data.get('flame', False)
    health_status = data.get('health_status', 'Healthy')
    
    # EMERGENCY OVERRIDE - FIRE DETECTED
    if flame:
        predictions = []
        for months in [1, 3, 6, 12]:
            predictions.append({
                'months': months,
                'projected_health_score': 0.0,
                'projected_status': 'Failure',
                'recommendation': '🔥 FIRE DETECTED! IMMEDIATE EMERGENCY ACTION REQUIRED!'
            })
        return predictions
    
    # Get current health score
    if 'cnn_prediction' in data and data['cnn_prediction']:
        current_score = data['cnn_prediction']['confidence']
        if health_status == 'Critical':
            current_score = min(current_score, 30)
            degradation = 8.0
        elif health_status == 'Warning':
            current_score = min(current_score, 60)
            degradation = 4.5
        elif health_status == 'Monitor':
            current_score = min(current_score, 80)
            degradation = 2.5
        else:
            degradation = 1.5
    else:
        current_score = data.get('health_confidence', 85)
        degradation = 2.5
    
    predictions = []
    for months in [1, 3, 6, 12]:
        projected_score = max(0, current_score - (degradation * months))
        
        if projected_score >= 85:
            status = "Healthy"
            recommendation = "Continue regular monitoring, routine maintenance OK"
        elif projected_score >= 70:
            status = "Monitor"
            recommendation = "Schedule preventive maintenance within 2 months"
        elif projected_score >= 50:
            status = "Warning"
            recommendation = "Schedule inspection and maintenance within 1 month"
        elif projected_score >= 30:
            status = "Critical"
            recommendation = "Immediate inspection required, reduce load"
        else:
            status = "Failure"
            recommendation = "Plan for transformer replacement urgently"
        
        predictions.append({
            'months': months,
            'projected_health_score': round(projected_score, 1),
            'projected_status': status,
            'recommendation': recommendation
        })
    
    return predictions

# ============================================
# DATA CLASS
# ============================================

class TransformerMonitor:
    def __init__(self):
        self.current_data = {
            'trial': 0,
            'primary_voltage': 230.0,
            'primary_current': 0,
            'primary_power_kw': 0,
            'primary_pf': 0.85,
            'secondary_voltage': 19.0,
            'secondary_current': 0,
            'secondary_power_kw': 0,
            'secondary_pf': 0.82,
            'temperature': 27.0,
            'humidity': 45.0,
            'efficiency': 85.0,
            'fault_overcurrent': False,
            'fault_overtemp': False,
            'flame': False,
            'health_status': 'Healthy',
            'health_confidence': 85.0,
            'cnn_prediction': None,
            'timestamp': ''
        }
        
        self.data_history = deque(maxlen=100)
        self.alerts = deque(maxlen=50)
        self.recommendations = deque(maxlen=20)
        self.running = True
        self.last_alert_time = {}
        
    def get_cnn_prediction(self, data):
        """Get CNN model prediction"""
        if cnn_predictor.is_loaded:
            try:
                prediction = cnn_predictor.predict(data)
                if prediction:
                    return prediction
            except Exception as e:
                print(f"CNN prediction failed: {e}")
        return None
    
    def rule_based_analysis(self, data):
        """Fallback rule-based health analysis"""
        flame = data.get('flame', False)
        
        if flame:
            return 'Critical', 100.0
        
        score = 0
        current = data.get('primary_current', 0)
        temp = data.get('temperature', 27)
        efficiency = data.get('efficiency', 85)
        humidity = data.get('humidity', 45)
        
        # Current scoring (0-40 points)
        if current < 25:
            score += 40
        elif current < 35:
            score += 30
        elif current < 45:
            score += 15
        else:
            score += 5
        
        # Temperature scoring (0-30 points) - Range 0-180°C
        if temp < 85:
            score += 30
        elif temp < 105:
            score += 20
        elif temp < 130:
            score += 10
        elif temp <= 180:
            score += 5
        else:
            score += 0
        
        # Efficiency scoring (0-20 points)
        if efficiency > 92:
            score += 20
        elif efficiency > 88:
            score += 15
        elif efficiency > 85:
            score += 10
        elif efficiency > 80:
            score += 5
        else:
            score += 0
        
        # Humidity penalty (0-10 points)
        if humidity > 85:
            score -= 10
        elif humidity > 75:
            score -= 5
        elif humidity > 65:
            score -= 2
        
        score = max(0, min(100, score))
        
        if score >= 85:
            return 'Healthy', score
        elif score >= 70:
            return 'Monitor', score
        elif score >= 50:
            return 'Warning', score
        else:
            return 'Critical', score
    
    def analyze_health(self, data):
        """Analyze health using CNN first, fallback to rule-based"""
        
        flame = data.get('flame', False)
        
        if flame:
            return 'Critical', 100.0, {'Critical': 100.0, 'Healthy': 0, 'Monitor': 0, 'Warning': 0}
        
        cnn_result = self.get_cnn_prediction(data)
        
        if cnn_result:
            status = cnn_result['predicted_status']
            confidence = cnn_result['confidence']
            probabilities = cnn_result['probabilities']
            print(f"🤖 CNN Prediction: {status} ({confidence:.1f}%)")
            return status, confidence, probabilities
        else:
            status, confidence = self.rule_based_analysis(data)
            probabilities = {
                'Critical': 0.8 if status == 'Critical' else 0.05,
                'Healthy': 0.8 if status == 'Healthy' else 0.05,
                'Monitor': 0.7 if status == 'Monitor' else 0.05,
                'Warning': 0.7 if status == 'Warning' else 0.05
            }
            return status, confidence, probabilities
    
    def process_hardware_data(self, json_data):
        """Process incoming JSON data from hardware via HTTP endpoint"""
        try:
            data = json_data if isinstance(json_data, dict) else json.loads(json_data)

            if 'primary_current' not in data:
                print("Invalid hardware data: missing primary_current")
                return False

            primary_current = float(data.get('primary_current', 0))
            temperature = float(data.get('temperature', 27))
            humidity = float(data.get('humidity', 45))
            efficiency = float(data.get('efficiency', 85))
            flame = bool(data.get('flame', False))
            fault_overcurrent = bool(data.get('fault_overcurrent', False))

            primary_voltage = float(data.get('primary_voltage', 230))
            primary_power_kw = float(data.get('primary_power_kw', (230 * primary_current * 0.85) / 1000))
            primary_pf = float(data.get('primary_pf', 0.85))
            secondary_voltage = float(data.get('secondary_voltage', 19))
            secondary_current = float(data.get('secondary_current', primary_current * (230 / 19) * 0.95))
            secondary_power_kw = float(data.get('secondary_power_kw', (19 * secondary_current * 0.82) / 1000))
            secondary_pf = float(data.get('secondary_pf', 0.82))

            fault_overtemp = temperature > 130

            analysis_data = {
                'primary_voltage': primary_voltage,
                'primary_current': primary_current,
                'primary_power_kw': primary_power_kw,
                'primary_pf': primary_pf,
                'secondary_voltage': secondary_voltage,
                'secondary_current': secondary_current,
                'secondary_power_kw': secondary_power_kw,
                'secondary_pf': secondary_pf,
                'temperature': temperature,
                'humidity': humidity,
                'efficiency': efficiency,
                'flame': flame
            }

            health_status, confidence, probabilities = self.analyze_health(analysis_data)

            self.current_data = {
                'trial': self.current_data.get('trial', 0) + 1,
                'primary_voltage': primary_voltage,
                'primary_current': primary_current,
                'primary_power_kw': primary_power_kw,
                'primary_pf': primary_pf,
                'secondary_voltage': secondary_voltage,
                'secondary_current': secondary_current,
                'secondary_power_kw': secondary_power_kw,
                'secondary_pf': secondary_pf,
                'temperature': temperature,
                'humidity': humidity,
                'efficiency': efficiency,
                'fault_overcurrent': fault_overcurrent,
                'fault_overtemp': fault_overtemp,
                'flame': flame,
                'health_status': health_status,
                'health_confidence': confidence,
                'health_probabilities': probabilities,
                'cnn_prediction': {
                    'predicted_status': health_status,
                    'confidence': confidence,
                    'probabilities': probabilities
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            self.generate_recommendations()
            self.check_alerts()
            self.data_history.append(self.current_data.copy())
            save_reading(self.current_data)

            life_data = calculate_life_expectancy(self.current_data)
            predictions = generate_predictions(self.current_data)

            socketio.emit('data_update', {
                'current': self.current_data,
                'life_expectancy': life_data,
                'predictions': predictions
            })

            cnn_status = "CNN" if cnn_predictor.is_loaded else "Rule"
            flame_marker = "🔥🔥🔥 FIRE! 🔥🔥🔥" if flame else ""
            print(f"📊 [{cnn_status} H/W] Trial {self.current_data['trial']}: I={primary_current:.1f}A "
                  f"T={temperature:.0f}C H={humidity:.0f}% "
                  f"Status={health_status} ({confidence:.0f}%) {flame_marker}")

            return True

        except Exception as e:
            print(f"Hardware data processing error: {e}")
            return False
    
    def generate_recommendations(self):
        """Generate recommendations based on CNN prediction and faults"""
        recommendations = []
        health = self.current_data.get('health_status', 'Healthy')
        current = self.current_data.get('primary_current', 0)
        temp = self.current_data.get('temperature', 27)
        humidity = self.current_data.get('humidity', 45)
        flame = self.current_data.get('flame', False)
        
        cnn_conf = self.current_data.get('cnn_prediction', {}).get('confidence', 0)
        
        # EMERGENCY: Fire - OVERRIDE EVERYTHING
        if flame:
            recommendations.append({
                'type': 'emergency',
                'message': '🔥🔥🔥 FIRE DETECTED! EMERGENCY PROTOCOL ACTIVATED! 🔥🔥🔥',
                'actions': [
                    '🚨 IMMEDIATE SHUTDOWN REQUIRED - DO NOT DELAY!',
                    '📞 Call emergency services (Fire Department) IMMEDIATELY',
                    '🏃 EVACUATE the area - Safety first!',
                    '🧯 Use CO2 or Dry Chemical extinguisher ONLY - NEVER use water',
                    '⚡ Isolate power source at main breaker',
                    '📋 Activate emergency response team'
                ],
                'priority': 'CRITICAL'
            })
            self.recommendations = deque(recommendations, maxlen=20)
            socketio.emit('recommendations_update', {'recommendations': list(self.recommendations)})
            return
        
        # High humidity alert
        if humidity > 85:
            recommendations.append({
                'type': 'critical',
                'message': f'💧💧 CRITICAL HUMIDITY: {humidity:.0f}% - Risk of insulation failure and flashover!',
                'actions': [
                    '🚨 Install industrial dehumidifier IMMEDIATELY',
                    '🔧 Check for water ingress and seal leaks',
                    '💨 Increase ventilation drastically',
                    '📊 Monitor insulation resistance hourly',
                    '⚠️ Reduce load if possible'
                ],
                'priority': 'HIGH'
            })
        elif humidity > 75:
            recommendations.append({
                'type': 'warning',
                'message': f'⚠️ HIGH HUMIDITY: {humidity:.0f}% - Monitoring required',
                'actions': [
                    'Install dehumidifier',
                    'Check for condensation',
                    'Increase ventilation',
                    'Monitor insulation resistance'
                ],
                'priority': 'MEDIUM'
            })
        
        # Overcurrent fault
        if self.current_data.get('fault_overcurrent', False):
            recommendations.append({
                'type': 'critical',
                'message': f'🔴 OVERCURRENT FAULT: {current:.1f}A exceeds 40A limit',
                'actions': [
                    'Reduce load by 50% immediately',
                    'Check for short circuits',
                    'Inspect all connected equipment',
                    'Schedule emergency maintenance'
                ],
                'priority': 'HIGH'
            })
        
        # Overtemperature fault - NOW USING 130°C THRESHOLD
        if temp > 130:
            recommendations.append({
                'type': 'critical',
                'message': f'🔥 HIGH TEMPERATURE: {temp:.1f}°C exceeds 130°C limit. Max range: 0-180°C',
                'actions': [
                    'Check cooling fans operation immediately',
                    'Clean radiator fins and ensure proper airflow',
                    'Reduce load by 50% or more',
                    'Inspect oil levels and quality',
                    '⚠️ Critical temperature - Monitor continuously'
                ],
                'priority': 'HIGH'
            })
        elif temp > 105:
            recommendations.append({
                'type': 'warning',
                'message': f'⚠️ ELEVATED TEMPERATURE: {temp:.1f}°C (Normal: 0-85°C, Warning: 105-130°C)',
                'actions': [
                    'Check cooling system operation',
                    'Monitor temperature trend',
                    'Consider load reduction',
                    'Schedule inspection'
                ],
                'priority': 'MEDIUM'
            })
        
        # Health-based recommendations
        if health == 'Critical':
            recommendations.append({
                'type': 'critical',
                'message': f'🚨 CRITICAL CONDITION - Immediate action required! (AI confidence: {cnn_conf:.1f}%)',
                'actions': [
                    '⚠️ Prepare for emergency shutdown',
                    '📞 Call maintenance team immediately',
                    '📊 Monitor every minute',
                    '🔄 Prepare backup transformer',
                    '📝 Document all readings'
                ],
                'priority': 'HIGH'
            })
        elif health == 'Warning':
            recommendations.append({
                'type': 'warning',
                'message': f'⚠️ Warning: Transformer showing signs of deterioration (AI confidence: {cnn_conf:.1f}%)',
                'actions': [
                    '📅 Schedule inspection within 24 hours',
                    '📉 Reduce load if possible',
                    '📈 Increase monitoring frequency',
                    '👂 Check for unusual sounds/vibrations'
                ],
                'priority': 'MEDIUM'
            })
        elif health == 'Monitor':
            recommendations.append({
                'type': 'monitor',
                'message': f'📊 Increased monitoring recommended (AI confidence: {cnn_conf:.1f}%)',
                'actions': [
                    '📝 Log readings every hour',
                    '⚖️ Check load balancing',
                    '🔧 Schedule maintenance next week',
                    '📈 Monitor temperature and humidity trends'
                ],
                'priority': 'LOW'
            })
        else:
            recommendations.append({
                'type': 'normal',
                'message': f'✅ Transformer operating normally (AI confidence: {cnn_conf:.1f}%)',
                'actions': [
                    '✅ Continue regular monitoring',
                    '📅 Routine maintenance on schedule',
                    '📝 Document all readings',
                    '👁️ Weekly visual inspection'
                ],
                'priority': 'NORMAL'
            })
        
        self.recommendations = deque(recommendations, maxlen=20)
        socketio.emit('recommendations_update', {'recommendations': list(self.recommendations)})
    
    def check_alerts(self):
        """Check and generate alerts"""
        alerts = []
        current_time = time.time()
        flame = self.current_data.get('flame', False)
        health = self.current_data.get('health_status', 'Healthy')
        humidity = self.current_data.get('humidity', 45)
        temp = self.current_data.get('temperature', 27)
        current = self.current_data.get('primary_current', 0)
        
        # FIRE alert
        if flame:
            if 'fire' not in self.last_alert_time or current_time - self.last_alert_time['fire'] > 10:
                alerts.append({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'type': 'emergency',
                    'message': '🔥🔥🔥 FIRE DETECTED! EMERGENCY SHUTDOWN REQUIRED! 🔥🔥🔥'
                })
                self.last_alert_time['fire'] = current_time
        
        # Critical humidity alert
        if humidity > 85:
            if 'high_humidity' not in self.last_alert_time or current_time - self.last_alert_time['high_humidity'] > 1800:
                alerts.append({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'type': 'critical',
                    'message': f'💧💧 CRITICAL HUMIDITY: {humidity:.0f}% - Risk of flashover!'
                })
                self.last_alert_time['high_humidity'] = current_time
        
        # Overcurrent alert
        if self.current_data.get('fault_overcurrent', False):
            if 'overcurrent' not in self.last_alert_time or current_time - self.last_alert_time['overcurrent'] > 60:
                alerts.append({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'type': 'critical',
                    'message': f"⚠️ OVERCURRENT: {current:.1f}A exceeds 40A limit"
                })
                self.last_alert_time['overcurrent'] = current_time
        
        # Overtemperature alert - UPDATED THRESHOLD
        if temp > 130:
            if 'overtemp' not in self.last_alert_time or current_time - self.last_alert_time['overtemp'] > 300:
                alerts.append({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'type': 'critical',
                    'message': f"🌡️ HIGH TEMPERATURE: {temp:.1f}°C exceeds 130°C - Check cooling! Operating range: 0-180°C"
                })
                self.last_alert_time['overtemp'] = current_time
        
        # Health status alert
        if health in ['Critical', 'Failure'] and not flame:
            if 'health' not in self.last_alert_time or current_time - self.last_alert_time['health'] > 120:
                alerts.append({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'type': 'critical',
                    'message': f"🚨 {health.upper()} HEALTH STATUS! Immediate attention required!"
                })
                self.last_alert_time['health'] = current_time
        
        for alert in alerts:
            self.alerts.append(alert)
            save_alert(alert)
            socketio.emit('new_alert', alert)
            # Send email for critical/emergency alerts
            if alert['type'] in ('emergency', 'critical'):
                send_email_alert(
                    f"🚨 Transformer Alert: {alert['type'].upper()}",
                    f"<h2>{alert['message']}</h2>"
                    f"<p><b>Time:</b> {alert['timestamp']}</p>"
                    f"<hr><p><b>Temperature:</b> {temp:.1f}°C</p>"
                    f"<p><b>Current:</b> {current:.1f}A</p>"
                    f"<p><b>Health:</b> {health}</p>"
                    f"<p><b>Humidity:</b> {humidity:.0f}%</p>"
                    f"<hr><p><small>Transformer Health Monitoring System</small></p>"
                )

# ============================================
# FLASK API ENDPOINTS
# ============================================

monitor = TransformerMonitor()

@app.route('/')
def index():
    return send_from_directory('.', 'dashboard.html')

@app.route('/dashboard.html')
def dashboard():
    return send_from_directory('.', 'dashboard.html')

@app.route('/api/current-data')
def get_current_data():
    life_data = calculate_life_expectancy(monitor.current_data)
    predictions = generate_predictions(monitor.current_data)
    
    return jsonify({
        'success': True,
        'current': monitor.current_data,
        'life_expectancy': life_data,
        'predictions': predictions,
        'cnn_loaded': cnn_predictor.is_loaded
    })

@app.route('/api/history')
def get_history():
    limit = 50
    history_list = list(monitor.data_history)[-limit:]
    return jsonify({
        'success': True,
        'data': history_list
    })

@app.route('/api/alerts')
def get_alerts():
    return jsonify({
        'success': True,
        'alerts': list(monitor.alerts)
    })

@app.route('/api/recommendations')
def get_recommendations():
    return jsonify({
        'success': True,
        'recommendations': list(monitor.recommendations),
        'cnn_loaded': cnn_predictor.is_loaded
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chatbot endpoint using DeepSeek API"""
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'response': 'Please ask a question.'})
    
    life_data = calculate_life_expectancy(monitor.current_data)
    predictions = generate_predictions(monitor.current_data)
    
    response = get_deepseek_response(user_message, monitor.current_data, life_data, predictions)
    
    return jsonify({'response': response})

@app.route('/api/cnn-info')
def get_cnn_info():
    if cnn_predictor.is_loaded:
        return jsonify({
            'loaded': True,
            'classes': list(cnn_predictor.label_encoder.classes_),
            'input_shape': cnn_predictor.model.input_shape
        })
    else:
        return jsonify({
            'loaded': False,
            'message': 'CNN model not loaded - using rule-based fallback'
        })

@app.route('/api/upload', methods=['POST'])
def upload_hardware_data():
    """Endpoint for hardware (ESP32/Arduino) to send sensor data via HTTP POST"""
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({'success': False, 'error': 'No JSON data received'}), 400

        success = monitor.process_hardware_data(json_data)
        if success:
            return jsonify({
                'success': True,
                'status': monitor.current_data['health_status'],
                'confidence': monitor.current_data['health_confidence']
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to process data'}), 400

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config')
def get_config():
    """Return public configuration (Mapbox token, etc.)"""
    return jsonify({
        'mapbox_token': os.environ.get('MAPBOX_TOKEN', ''),
        'cnn_loaded': cnn_predictor.is_loaded
    })

@app.route('/api/upload/mock', methods=['GET'])
def upload_mock_endpoint():
    """Simple GET endpoint to test hardware connectivity"""
    return jsonify({
        'success': True,
        'message': 'Hardware data upload endpoint is ready',
        'instructions': 'Send POST request with JSON body containing sensor readings'
    })

# ============================================
# DATABASE HISTORY ENDPOINT
# ============================================

@app.route('/api/history/db')
def get_history_db():
    limit = request.args.get('limit', 100, type=int)
    rows = get_readings_from_db(limit)
    return jsonify({'success': True, 'data': rows})

@app.route('/api/alerts/db')
def get_alerts_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 50").fetchall()
        conn.close()
        alerts = [{'id': r[0], 'timestamp': r[1], 'type': r[2], 'message': r[3]} for r in rows]
        return jsonify({'success': True, 'alerts': alerts})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# CSV EXPORT
# ============================================

@app.route('/api/export/csv')
def export_csv():
    try:
        limit = request.args.get('limit', 10000, type=int)
        rows = get_readings_from_db(limit)
        lines = ["timestamp,primary_current,primary_voltage,primary_power_kw,primary_pf,"
                 "secondary_current,secondary_voltage,secondary_power_kw,secondary_pf,"
                 "temperature,humidity,efficiency,fault_overcurrent,fault_overtemp,flame,"
                 "health_status,health_confidence"]
        for r in reversed(rows):
            lines.append(
                f"{r['timestamp']},{r['primary_current']},{r['primary_voltage']},"
                f"{r['primary_power_kw']},{r['primary_pf']},{r['secondary_current']},"
                f"{r['secondary_voltage']},{r['secondary_power_kw']},{r['secondary_pf']},"
                f"{r['temperature']},{r['humidity']},{r['efficiency']},"
                f"{r['fault_overcurrent']},{r['fault_overtemp']},{r['flame']},"
                f"{r['health_status']},{r['health_confidence']}"
            )
        csv_str = "\n".join(lines)
        return app.response_class(
            csv_str,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=transformer_data_{datetime.now().strftime("%Y%m%d")}.csv'}
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# AGGREGATED STATISTICS
# ============================================

def get_stats(group_by):
    """group_by: 'daily', 'weekly', or 'monthly'"""
    try:
        if group_by == 'daily':
            date_fmt = "%Y-%m-%d"
            trunc = "substr(timestamp, 1, 10)"
        elif group_by == 'weekly':
            date_fmt = "%Y-%W"
            trunc = "strftime('%Y-%W', timestamp)"
        else:
            date_fmt = "%Y-%m"
            trunc = "substr(timestamp, 1, 7)"

        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(f"""
            SELECT
                {trunc} as period,
                COUNT(*) as count,
                ROUND(AVG(temperature), 1) as temp_avg,
                ROUND(MIN(temperature), 1) as temp_min,
                ROUND(MAX(temperature), 1) as temp_max,
                ROUND(AVG(primary_current), 1) as current_avg,
                ROUND(MIN(primary_current), 1) as current_min,
                ROUND(MAX(primary_current), 1) as current_max,
                ROUND(AVG(humidity), 1) as humidity_avg,
                ROUND(MIN(humidity), 1) as humidity_min,
                ROUND(MAX(humidity), 1) as humidity_max,
                ROUND(AVG(health_confidence), 1) as health_avg,
                ROUND(AVG(efficiency), 1) as efficiency_avg
            FROM readings
            GROUP BY period
            ORDER BY period DESC
            LIMIT 90
        """).fetchall()
        conn.close()

        cols = ['period','count','temp_avg','temp_min','temp_max',
                'current_avg','current_min','current_max',
                'humidity_avg','humidity_min','humidity_max',
                'health_avg','efficiency_avg']
        return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        print(f"Stats error: {e}")
        return []

@app.route('/api/stats/daily')
def stats_daily():
    return jsonify({'success': True, 'stats': get_stats('daily')})

@app.route('/api/stats/weekly')
def stats_weekly():
    return jsonify({'success': True, 'stats': get_stats('weekly')})

@app.route('/api/stats/monthly')
def stats_monthly():
    return jsonify({'success': True, 'stats': get_stats('monthly')})

# ============================================
# PDF REPORT GENERATION
# ============================================

@app.route('/api/report')
def generate_report():
    try:
        from fpdf import FPDF

        data = monitor.current_data
        life = calculate_life_expectancy(data)
        preds = generate_predictions(data)
        readings = get_readings_from_db(20)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "Transformer Health Report", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)

        # Health Summary
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(16, 185, 129)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 9, "  HEALTH SUMMARY", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        flame = data.get('flame', False)
        status = data.get('health_status', 'Unknown')
        conf = data.get('health_confidence', 0)

        if flame:
            status_color = (239, 68, 68)
        elif status == 'Healthy':
            status_color = (16, 185, 129)
        elif status == 'Monitor':
            status_color = (245, 158, 11)
        elif status == 'Warning':
            status_color = (249, 115, 22)
        else:
            status_color = (239, 68, 68)

        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(*status_color)
        label = "🔥 FIRE DETECTED!" if flame else f"{status} ({conf:.0f}%)"
        pdf.cell(0, 14, f"  {label}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

        # Sensor Readings Table
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(59, 130, 246)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 9, "  CURRENT READINGS", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        def row(label, value, unit):
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(80, 7, f"  {label}", border=1)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, f"{value} {unit}", border=1, new_x="LMARGIN", new_y="NEXT")

        row("Primary Current", f"{data.get('primary_current', 0):.1f}", "A")
        row("Primary Voltage", f"{data.get('primary_voltage', 230):.1f}", "V")
        row("Primary Power", f"{data.get('primary_power_kw', 0):.2f}", "kW")
        row("Power Factor", f"{data.get('primary_pf', 0):.3f}", "")
        row("Secondary Current", f"{data.get('secondary_current', 0):.1f}", "A")
        row("Secondary Voltage", f"{data.get('secondary_voltage', 19):.2f}", "V")
        row("Temperature", f"{data.get('temperature', 27):.1f}", "C")
        row("Humidity", f"{data.get('humidity', 45):.1f}", "%")
        row("Efficiency", f"{data.get('efficiency', 85):.1f}", "%")
        row("Health Confidence", f"{conf:.1f}", "%")
        pdf.ln(4)

        # Faults
        faults = []
        if data.get('fault_overcurrent'): faults.append("Overcurrent")
        if data.get('fault_overtemp'): faults.append("Overtemp")
        if data.get('flame'): faults.append("FIRE")
        fault_str = ", ".join(faults) if faults else "None"
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(80, 7, "  Active Faults:", border=1)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, f"  {fault_str}", border=1, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # Life Expectancy
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(139, 92, 246)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 9, "  LIFE EXPECTANCY", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        row("Health Score", f"{life.get('health_score', 0):.1f}", "%")
        row("Remaining Life", f"{life.get('remaining_years', 0):.1f}", "years")
        row("Degradation Rate", f"{life.get('degradation_rate', 0):.1f}", "%")
        row("Est. Failure", life.get('estimated_failure_date', 'N/A'), "")
        pdf.ln(4)

        # Predictions
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(245, 158, 11)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 9, "  FUTURE PREDICTIONS", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        for p in preds:
            row(f"{p['months']} Month{'s' if p['months'] > 1 else ''}",
                f"{p['projected_health_score']:.0f}% - {p['projected_status']}", "")
        pdf.ln(4)

        # Recent History
        if readings:
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_fill_color(107, 114, 128)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 9, "  RECENT HISTORY (last 20 readings)", fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 7)
            for r in reversed(readings):
                pdf.cell(0, 4,
                    f"  {r.get('timestamp','')[:19]}  |  I={r.get('primary_current',0):.1f}A  "
                    f"T={r.get('temperature',0):.0f}C  H={r.get('humidity',0):.0f}%  "
                    f"Status={r.get('health_status','')}",
                    new_x="LMARGIN", new_y="NEXT")

        # Output PDF as response
        pdf_bytes = pdf.output()
        response = app.response_class(
            pdf_bytes,
            mimetype='application/pdf',
            headers={'Content-Disposition': f'attachment; filename=transformer_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'}
        )
        return response

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# SIMULATION MODE (for development only)
# ============================================

def simulate_data():
    """Simulate data for testing"""
    trial = 0
    angle = 0
    
    print("\n📊 SIMULATION MODE ACTIVE - Generating test data...\n")
    
    while monitor.running:
        trial += 1
        angle += 0.05
        
        primary_current = 27.5 + 22.5 * np.sin(angle)
        primary_current += random.uniform(-2, 2)
        primary_current = max(5, min(55, primary_current))
        
        temperature = 27 + (primary_current / 55) * 180  # Up to 180°C max
        temperature += random.uniform(-5, 5)
        temperature = max(0, min(180, temperature))
        
        humidity = 45 + 30 * np.sin(angle / 2) + random.uniform(-10, 10)
        humidity = max(20, min(95, humidity))
        
        fault_overcurrent = primary_current > 40
        fault_overtemp = temperature > 130  # Now triggers at 130°C
        flame = random.random() < 0.02
        
        efficiency = 85 + random.uniform(-5, 5)
        efficiency = max(75, min(95, efficiency))
        
        mock_data = {
            'trial': trial,
            'primary_voltage': 230,
            'primary_current': primary_current,
            'primary_power_kw': (230 * primary_current * 0.85) / 1000,
            'primary_pf': 0.85,
            'secondary_voltage': 19,
            'secondary_current': primary_current * (230 / 19) * 0.95,
            'secondary_power_kw': (19 * primary_current * (230 / 19) * 0.95 * 0.82) / 1000,
            'secondary_pf': 0.82,
            'temperature': temperature,
            'humidity': humidity,
            'efficiency': efficiency,
            'fault_overcurrent': fault_overcurrent,
            'fault_overtemp': fault_overtemp,
            'flame': flame,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        health_status, confidence, probabilities = monitor.analyze_health(mock_data)
        mock_data['health_status'] = health_status
        mock_data['health_confidence'] = confidence
        mock_data['health_probabilities'] = probabilities
        mock_data['cnn_prediction'] = {
            'predicted_status': health_status,
            'confidence': confidence,
            'probabilities': probabilities
        }
        
        monitor.current_data = mock_data
        monitor.generate_recommendations()
        monitor.check_alerts()
        monitor.data_history.append(mock_data.copy())
        save_reading(mock_data)
        
        life_data = calculate_life_expectancy(monitor.current_data)
        predictions = generate_predictions(monitor.current_data)
        
        socketio.emit('data_update', {
            'current': monitor.current_data,
            'life_expectancy': life_data,
            'predictions': predictions
        })
        
        cnn_status = "CNN" if cnn_predictor.is_loaded else "Rule"
        flame_marker = "🔥🔥🔥 FIRE! 🔥🔥🔥" if flame else ""
        print(f"📊 [{cnn_status} SIM] Trial {trial}: I={primary_current:.1f}A "
              f"T={temperature:.0f}C H={humidity:.0f}% Status={health_status} ({confidence:.0f}%) "
              f"{flame_marker}")
        
        time.sleep(1)

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == '__main__':
    print("="*60)
    print("TRANSFORMER HEALTH MONITORING SYSTEM")
    print("CNN INTEGRATED VERSION - FULLY CORRECTED")
    print("="*60)
    
    if cnn_predictor.is_loaded:
        print("\n🤖 CNN MODEL STATUS: LOADED AND READY")
        print(f"   Number of classes: {len(cnn_predictor.label_encoder.classes_)}")
    else:
        print("\n⚠️ CNN MODEL NOT LOADED - Using rule-based fallback")
    
    print("\n🌍 LOCATION: Lusaka, Zambia (-15.3875, 28.0473)")
    print("🗺️ Map: Satellite view with precise transformer marker")
    print("🌡️ Temperature Range: 0°C to 180°C")
    print("   - Normal: 0-85°C")
    print("   - Elevated: 85-105°C")
    print("   - High: 105-130°C")
    print("   - Critical: 130-180°C")
    init_db()
    cleanup_old_records()
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    print("\n📡 Mode: Receiving data from HARDWARE via HTTP endpoint")
    print("   POST sensor data to: /api/upload")
    
    # Start simulation mode only if SIMULATION_MODE env var is set
    if os.environ.get('SIMULATION_MODE', '').lower() in ('true', '1', 'yes'):
        print("\n🧪 SIMULATION MODE ENABLED - Generating test data...")
        reader_thread = threading.Thread(target=simulate_data, daemon=True)
        reader_thread.start()
    else:
        print("   ⏳ Waiting for hardware data on /api/upload...")
    
    print("\n✅ System started!")
    print("📊 Open: http://localhost:5000")
    print("🔌 WebSocket: Real-time updates enabled")
    print("🤖 Chatbot: Powered by DeepSeek AI")
    print("\nPress Ctrl+C to stop\n")
    
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)