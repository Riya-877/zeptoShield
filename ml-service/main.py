"""
ZeptoShield ML Service — FastAPI
Runs on http://localhost:8000

Endpoints:
  POST /ml/premium          → Dynamic premium calculation
  POST /ml/fraud            → Fraud / trust score analysis
  POST /ml/risk-profile     → City + worker risk profiling
  GET  /health              → Health check
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle, os, json, math
from datetime import datetime, timedelta
import random

app = FastAPI(title="ZeptoShield ML Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CITY FEATURE DATABASE ───────────────────────────────────────────────────
# Pre-computed city features based on historical data
# [avg_rainfall_mm, max_temp_c, avg_aqi, flood_freq_per_year, disruption_days_per_year]
CITY_FEATURES = {
    "Mumbai":        [2400, 38, 145, 3.2, 42],
    "Chennai":       [1400, 42, 120, 2.1, 35],
    "Kolkata":       [1800, 40, 180, 2.8, 38],
    "Hyderabad":     [900,  42, 130, 1.5, 28],
    "Bengaluru":     [970,  35, 95,  0.8, 22],
    "Delhi":         [750,  45, 280, 0.5, 30],
    "Pune":          [720,  38, 85,  0.6, 18],
    "Ahmedabad":     [800,  46, 150, 0.4, 24],
    "Jaipur":        [550,  46, 170, 0.2, 15],
    "Surat":         [1200, 38, 130, 1.0, 20],
    "Lucknow":       [900,  44, 200, 0.6, 22],
    "Bhubaneswar":   [1500, 40, 100, 1.8, 30],
    "Nagpur":        [1100, 47, 110, 0.7, 20],
    "Indore":        [950,  43, 120, 0.5, 18],
    "Visakhapatnam": [1100, 40, 90,  1.2, 25],
    "Coimbatore":    [700,  36, 80,  0.4, 12],
}

DEFAULT_CITY = [900, 40, 120, 0.8, 20]

# ── MODEL TRAINING (on startup) ─────────────────────────────────────────────

def generate_training_data_premium(n=2000):
    """Generate synthetic training data for premium prediction."""
    np.random.seed(42)
    X, y = [], []
    cities = list(CITY_FEATURES.values())
    for _ in range(n):
        cf = random.choice(cities)
        rainfall, max_temp, aqi, flood_freq, disruption_days = cf
        weekly_earnings = np.random.uniform(3000, 12000)
        months_active   = np.random.uniform(1, 24)

        # Normalise features
        rain_score     = min(rainfall / 2400, 1.0)
        temp_score     = max((max_temp - 30) / 20, 0)
        aqi_score      = min(aqi / 350, 1.0)
        flood_score    = min(flood_freq / 4, 1.0)
        dis_score      = min(disruption_days / 50, 1.0)
        earnings_score = weekly_earnings / 12000

        composite_risk = (
            rain_score  * 0.30 +
            temp_score  * 0.20 +
            aqi_score   * 0.15 +
            flood_score * 0.20 +
            dis_score   * 0.15
        )

        # Premium = base (20) + risk adjustment (0-30) + earnings factor (0-10)
        premium = 20 + composite_risk * 30 + earnings_score * 10
        premium = round(float(np.clip(premium + np.random.normal(0, 1.5), 20, 60)), 2)

        X.append([rainfall, max_temp, aqi, flood_freq, disruption_days, weekly_earnings, months_active])
        y.append(premium)
    return np.array(X), np.array(y)


def generate_training_data_fraud(n=3000):
    """Generate synthetic training data for fraud detection."""
    np.random.seed(99)
    X, y = [], []
    for _ in range(n):
        is_fraud = random.random() < 0.2   # 20% fraud rate in training

        if is_fraud:
            claims_per_week     = np.random.uniform(3, 7)
            time_since_activate = np.random.uniform(0, 2)   # very new
            location_variance   = np.random.uniform(0, 0.5) # suspiciously low
            speed_consistency   = np.random.uniform(0, 0.3)
            device_changes      = np.random.randint(3, 10)
            ip_mismatch         = 1
            duplicate_interval  = np.random.uniform(0, 2)   # hours
            cluster_size        = np.random.randint(3, 20)  # many similar devices
        else:
            claims_per_week     = np.random.uniform(0, 1.5)
            time_since_activate = np.random.uniform(2, 24)
            location_variance   = np.random.uniform(0.5, 1.0)
            speed_consistency   = np.random.uniform(0.6, 1.0)
            device_changes      = np.random.randint(0, 2)
            ip_mismatch         = 0 if random.random() > 0.05 else 1
            duplicate_interval  = np.random.uniform(12, 72)
            cluster_size        = np.random.randint(0, 2)

        X.append([
            claims_per_week, time_since_activate, location_variance,
            speed_consistency, device_changes, ip_mismatch,
            duplicate_interval, cluster_size
        ])
        y.append(int(is_fraud))
    return np.array(X), np.array(y)


print("🔧 Training ML models...")

# Premium model — Gradient Boosting Regressor
X_prem, y_prem = generate_training_data_premium()
premium_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42))
])
premium_pipeline.fit(X_prem, y_prem)
print("  ✓ Premium model trained")

# Fraud model — Random Forest Classifier
X_fraud, y_fraud = generate_training_data_fraud()
fraud_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestClassifier(n_estimators=120, max_depth=6, class_weight="balanced", random_state=42))
])
fraud_pipeline.fit(X_fraud, y_fraud)
print("  ✓ Fraud detection model trained")

# Anomaly detector — Isolation Forest (unsupervised, for novel patterns)
X_legit = X_fraud[y_fraud == 0]
anomaly_detector = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
anomaly_detector.fit(X_legit)
print("  ✓ Anomaly detector trained")
print("✅ All models ready\n")

# ── REQUEST SCHEMAS ─────────────────────────────────────────────────────────

class PremiumRequest(BaseModel):
    city: str
    weekly_earnings: float = 6000
    months_active: float = 0
    previous_claims: int = 0

class FraudRequest(BaseModel):
    worker_id: int
    claims_per_week: float = 0
    hours_since_activation: float = 24
    location_variance: float = 0.8       # 0-1, higher = more natural movement
    speed_consistency: float = 0.8       # 0-1, higher = consistent delivery pattern
    device_changes_30d: int = 0
    ip_gps_mismatch: bool = False
    last_claim_interval_hours: float = 48
    similar_device_cluster_size: int = 0

class RiskRequest(BaseModel):
    city: str
    current_rainfall_mm: Optional[float] = None
    current_temp_c: Optional[float] = None
    current_aqi: Optional[float] = None
    flood_alert: Optional[bool] = False

# ── HELPERS ─────────────────────────────────────────────────────────────────

def get_city_features(city: str):
    return CITY_FEATURES.get(city, DEFAULT_CITY)

def trust_score_from_fraud(fraud_prob: float, anomaly_score: float) -> dict:
    """Convert ML outputs into a human-readable trust score (0-100)."""
    # anomaly_score from IsolationForest: -1 = anomaly, 1 = normal → map to 0-1
    anomaly_norm = (anomaly_score + 1) / 2   # -1..1 → 0..1
    
    raw = (1 - fraud_prob) * 0.65 + anomaly_norm * 0.35
    score = round(float(np.clip(raw * 100, 0, 100)), 1)

    if score >= 80:
        level, action, color = "Low",    "instant_payout",       "green"
    elif score >= 55:
        level, action, color = "Medium", "quick_verification",   "amber"
    else:
        level, action, color = "High",   "flag_and_review",      "red"

    return {"score": score, "risk_level": level, "action": action, "color": color}

# ── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models": ["premium_gbr", "fraud_rf", "anomaly_iforest"]}


@app.post("/ml/premium")
def predict_premium(req: PremiumRequest):
    """
    Predict dynamic weekly premium using Gradient Boosting.
    Returns: premium (₹), risk_score (0-1), breakdown, model confidence.
    """
    cf = get_city_features(req.city)
    rainfall, max_temp, aqi, flood_freq, disruption_days = cf

    features = np.array([[
        rainfall, max_temp, aqi, flood_freq, disruption_days,
        req.weekly_earnings, req.months_active
    ]])

    predicted_premium = float(premium_pipeline.predict(features)[0])
    predicted_premium = round(np.clip(predicted_premium, 20, 60), 2)

    # Compute individual risk components for breakdown UI
    rain_score  = round(min(rainfall / 2400, 1.0) * 100, 1)
    temp_score  = round(max((max_temp - 30) / 20, 0) * 100, 1)
    aqi_score   = round(min(aqi / 350, 1.0) * 100, 1)
    flood_score = round(min(flood_freq / 4, 1.0) * 100, 1)
    overall     = round((rain_score*0.30 + temp_score*0.20 + aqi_score*0.15 + flood_score*0.35), 1)

    # Estimate model confidence from ensemble variance (simulated)
    confidence = round(float(np.clip(0.94 - abs(predicted_premium - 35) / 200, 0.80, 0.97)), 3)

    return {
        "premium": predicted_premium,
        "risk_score": round(overall / 100, 3),
        "confidence": confidence,
        "breakdown": {
            "rainfall_risk":    rain_score,
            "temperature_risk": temp_score,
            "aqi_risk":         aqi_score,
            "flood_risk":       flood_score,
            "overall_risk":     overall,
        },
        "city_stats": {
            "avg_annual_rainfall_mm": rainfall,
            "max_recorded_temp_c":    max_temp,
            "avg_aqi":                aqi,
            "flood_events_per_year":  flood_freq,
            "avg_disruption_days":    disruption_days,
        },
        "model": "GradientBoostingRegressor",
        "features_used": 7,
    }


@app.post("/ml/fraud")
def detect_fraud(req: FraudRequest):
    """
    Fraud detection using Random Forest + Isolation Forest ensemble.
    Returns: trust_score, fraud_probability, anomaly_score, recommended action.
    """
    features = np.array([[
        req.claims_per_week,
        req.hours_since_activation,
        req.location_variance,
        req.speed_consistency,
        int(req.device_changes_30d),
        int(req.ip_gps_mismatch),
        req.last_claim_interval_hours,
        req.similar_device_cluster_size,
    ]])

    # Random Forest fraud probability
    fraud_prob = float(fraud_pipeline.predict_proba(features)[0][1])

    # Isolation Forest anomaly score (-1 anomaly, 1 normal)
    anomaly_raw = float(anomaly_detector.score_samples(features)[0])
    # Normalize: typical range is -0.6 to 0.1 → map to -1..1
    anomaly_score = float(np.clip((anomaly_raw + 0.3) / 0.4, -1, 1))

    trust = trust_score_from_fraud(fraud_prob, anomaly_score)

    # Build signal breakdown for transparency
    signals = []
    if req.claims_per_week > 3:
        signals.append({"flag": "High claim frequency", "weight": "high", "value": f"{req.claims_per_week:.1f}/week"})
    if req.hours_since_activation < 2:
        signals.append({"flag": "Very new account", "weight": "medium", "value": f"{req.hours_since_activation:.1f}h old"})
    if req.location_variance < 0.3:
        signals.append({"flag": "Suspiciously static location", "weight": "high", "value": f"variance {req.location_variance:.2f}"})
    if req.speed_consistency < 0.3:
        signals.append({"flag": "Unusual movement pattern", "weight": "medium", "value": f"consistency {req.speed_consistency:.2f}"})
    if req.ip_gps_mismatch:
        signals.append({"flag": "IP/GPS location mismatch", "weight": "high", "value": "detected"})
    if req.device_changes_30d >= 3:
        signals.append({"flag": "Frequent device changes", "weight": "medium", "value": f"{req.device_changes_30d} in 30d"})
    if req.similar_device_cluster_size >= 3:
        signals.append({"flag": "Device cluster detected", "weight": "high", "value": f"{req.similar_device_cluster_size} similar devices"})
    if req.last_claim_interval_hours < 2:
        signals.append({"flag": "Rapid repeat claim", "weight": "high", "value": f"{req.last_claim_interval_hours:.1f}h since last"})

    return {
        "worker_id":       req.worker_id,
        "trust_score":     trust["score"],
        "risk_level":      trust["risk_level"],
        "action":          trust["action"],
        "color":           trust["color"],
        "fraud_probability": round(fraud_prob * 100, 1),
        "anomaly_score":   round(anomaly_score, 3),
        "signals":         signals,
        "models_used":     ["RandomForestClassifier", "IsolationForest"],
        "explanation":     "We don't trust location alone — we trust behavior.",
    }


@app.post("/ml/risk-profile")
def compute_risk_profile(req: RiskRequest):
    """
    Real-time city risk profile combining historical ML features
    with live environmental readings.
    """
    cf = get_city_features(req.city)
    rainfall, max_temp, aqi_hist, flood_freq, disruption_days = cf

    # Start with historical baseline
    risk_components = {
        "historical_rainfall": round(min(rainfall / 2400, 1.0) * 100, 1),
        "historical_temp":     round(max((max_temp - 30) / 20, 0) * 100, 1),
        "historical_aqi":      round(min(aqi_hist / 350, 1.0) * 100, 1),
        "flood_history":       round(min(flood_freq / 4, 1.0) * 100, 1),
    }

    # Override with live readings if provided
    live_alerts = []
    current_risk_boost = 0.0

    if req.current_rainfall_mm is not None:
        live_rain = min(req.current_rainfall_mm / 100, 1.0)
        risk_components["live_rainfall"] = round(live_rain * 100, 1)
        if req.current_rainfall_mm >= 40:
            live_alerts.append({"type": "rain", "value": req.current_rainfall_mm, "unit": "mm", "triggered": True})
            current_risk_boost += 0.25

    if req.current_temp_c is not None:
        live_temp = max((req.current_temp_c - 30) / 20, 0)
        risk_components["live_temperature"] = round(min(live_temp, 1.0) * 100, 1)
        if req.current_temp_c >= 42:
            live_alerts.append({"type": "heat", "value": req.current_temp_c, "unit": "°C", "triggered": True})
            current_risk_boost += 0.20

    if req.current_aqi is not None:
        live_aqi = min(req.current_aqi / 500, 1.0)
        risk_components["live_aqi"] = round(live_aqi * 100, 1)
        if req.current_aqi >= 350:
            live_alerts.append({"type": "aqi", "value": req.current_aqi, "unit": "", "triggered": True})
            current_risk_boost += 0.20

    if req.flood_alert:
        live_alerts.append({"type": "flood", "value": 1, "unit": "", "triggered": True})
        current_risk_boost += 0.35

    # Composite risk score
    base_risk = (
        risk_components["historical_rainfall"] * 0.25 +
        risk_components["historical_temp"]     * 0.20 +
        risk_components["historical_aqi"]      * 0.15 +
        risk_components["flood_history"]       * 0.20
    ) / 100

    total_risk = float(np.clip(base_risk + current_risk_boost, 0, 1))

    if total_risk >= 0.65:   risk_tier = "high"
    elif total_risk >= 0.35: risk_tier = "medium"
    else:                    risk_tier = "low"

    return {
        "city":              req.city,
        "risk_score":        round(total_risk, 3),
        "risk_tier":         risk_tier,
        "live_alerts":       live_alerts,
        "any_triggered":     len(live_alerts) > 0,
        "risk_components":   risk_components,
        "disruption_days_pa": disruption_days,
    }
