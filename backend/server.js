/**
 * ZeptoShield Backend — Node.js + Express
 * Port: 5000
 *
 * Integrations:
 *   - Python ML service  → http://localhost:8000
 *   - Open-Meteo API     → free, no key needed (weather + AQI)
 *   - India Flood data   → NDMA public feed
 */

const express = require("express");
const cors    = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

const ML_URL     = process.env.ML_URL     || "http://localhost:8000";
const WEATHER_URL = "https://api.open-meteo.com/v1/forecast";
const AQI_URL     = "https://air-quality-api.open-meteo.com/v1/air-quality";

// ── IN-MEMORY DB ────────────────────────────────────────────────────────────
const DB = { workers: [], policies: [], claims: [], _id: 100 };
const uid = () => DB._id++;

// City → lat/lng for weather APIs
const CITY_COORDS = {
  Mumbai:        { lat: 19.076, lon: 72.877 },
  Chennai:       { lat: 13.082, lon: 80.270 },
  Kolkata:       { lat: 22.572, lon: 88.363 },
  Hyderabad:     { lat: 17.385, lon: 78.486 },
  Bengaluru:     { lat: 12.971, lon: 77.594 },
  Delhi:         { lat: 28.614, lon: 77.209 },
  Pune:          { lat: 18.520, lon: 73.856 },
  Ahmedabad:     { lat: 23.022, lon: 72.571 },
  Jaipur:        { lat: 26.912, lon: 75.787 },
  Surat:         { lat: 21.170, lon: 72.831 },
  Lucknow:       { lat: 26.850, lon: 80.949 },
  Bhubaneswar:   { lat: 20.296, lon: 85.824 },
  Nagpur:        { lat: 21.145, lon: 79.088 },
  Indore:        { lat: 22.719, lon: 75.857 },
  Visakhapatnam: { lat: 17.686, lon: 83.218 },
  Coimbatore:    { lat: 11.016, lon: 76.955 },
};

// ── ML SERVICE CALLER ───────────────────────────────────────────────────────
async function callML(endpoint, body) {
  try {
    const res = await fetch(`${ML_URL}${endpoint}`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
      signal:  AbortSignal.timeout(5000),
    });
    if (!res.ok) throw new Error(`ML service ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn(`ML service unavailable (${endpoint}):`, err.message);
    return null;
  }
}

// ── WEATHER + AQI FETCHER ───────────────────────────────────────────────────
async function fetchLiveWeather(city) {
  const coords = CITY_COORDS[city];
  if (!coords) return null;

  try {
    const [weatherRes, aqiRes] = await Promise.all([
      fetch(
        `${WEATHER_URL}?latitude=${coords.lat}&longitude=${coords.lon}` +
        `&current=temperature_2m,precipitation,rain,weathercode` +
        `&hourly=precipitation&forecast_days=1`,
        { signal: AbortSignal.timeout(4000) }
      ),
      fetch(
        `${AQI_URL}?latitude=${coords.lat}&longitude=${coords.lon}` +
        `&current=pm10,pm2_5,us_aqi&hourly=us_aqi&forecast_days=1`,
        { signal: AbortSignal.timeout(4000) }
      ),
    ]);

    const weather = weatherRes.ok ? await weatherRes.json() : null;
    const aqiData = aqiRes.ok    ? await aqiRes.json()    : null;

    // Accumulate today's rainfall from hourly data
    const hourlyRain = weather?.hourly?.precipitation || [];
    const totalRainToday = hourlyRain.slice(0, 24).reduce((a, b) => a + (b || 0), 0);

    return {
      temperature_c:    weather?.current?.temperature_2m  ?? null,
      rainfall_mm:      Math.round(totalRainToday * 10) / 10,
      current_rain_mm:  weather?.current?.rain             ?? 0,
      weather_code:     weather?.current?.weathercode      ?? null,
      aqi:              aqiData?.current?.us_aqi            ?? null,
      pm25:             aqiData?.current?.pm2_5             ?? null,
      pm10:             aqiData?.current?.pm10              ?? null,
      source:           "open-meteo",
      fetched_at:       new Date().toISOString(),
    };
  } catch (err) {
    console.warn("Weather API error:", err.message);
    return null;
  }
}

// ── RISK SCORE HELPER (fallback if ML down) ─────────────────────────────────
function fallbackRisk(city) {
  const high = ["Mumbai","Chennai","Kolkata","Hyderabad"];
  const med  = ["Bengaluru","Delhi","Pune","Ahmedabad"];
  if (high.includes(city)) return { tier: "high",   score: 0.75, premium: 40 };
  if (med.includes(city))  return { tier: "medium", score: 0.50, premium: 30 };
  return                          { tier: "low",    score: 0.25, premium: 20 };
}

// ── ROUTES ──────────────────────────────────────────────────────────────────

// GET live weather + AQI for a city
app.get("/api/weather/:city", async (req, res) => {
  const city = req.params.city;
  const weather = await fetchLiveWeather(city);
  if (!weather) return res.status(503).json({ error: "Weather data unavailable", city });

  // Get ML risk profile with live readings
  const mlProfile = await callML("/ml/risk-profile", {
    city,
    current_rainfall_mm: weather.rainfall_mm,
    current_temp_c:      weather.temperature_c,
    current_aqi:         weather.aqi,
    flood_alert:         false,
  });

  res.json({ city, weather, ml_risk_profile: mlProfile });
});

// POST register worker — calls ML for dynamic premium
app.post("/api/workers/register", async (req, res) => {
  const { name, phone, city, weeklyEarnings } = req.body;
  if (!name || !phone || !city)
    return res.status(400).json({ error: "name, phone and city are required" });

  const existing = DB.workers.find(w => w.phone === phone);
  if (existing) return res.status(409).json({ error: "Already registered", worker: existing });

  // Call ML service for dynamic premium
  const mlPremium = await callML("/ml/premium", {
    city,
    weekly_earnings: parseFloat(weeklyEarnings) || 6000,
    months_active: 0,
    previous_claims: 0,
  });

  const fb = fallbackRisk(city);
  const premium   = mlPremium ? Math.round(mlPremium.premium)       : fb.premium;
  const riskScore = mlPremium ? mlPremium.risk_score                 : fb.score;
  const riskTier  = mlPremium ? (riskScore >= 0.6 ? "high" : riskScore >= 0.35 ? "medium" : "low") : fb.tier;

  const worker = {
    id: uid(), name, phone, city,
    weeklyEarnings: parseFloat(weeklyEarnings) || 6000,
    premium, riskScore, riskTier,
    mlBreakdown: mlPremium?.breakdown || null,
    registeredAt: new Date().toISOString(),
  };

  DB.workers.push(worker);
  res.status(201).json({ message: "Registered successfully", worker, ml_used: !!mlPremium });
});

// GET worker + policy + claims
app.get("/api/workers/:phone", (req, res) => {
  const worker = DB.workers.find(w => w.phone === req.params.phone);
  if (!worker) return res.status(404).json({ error: "Worker not found" });
  const policy = DB.policies.find(p => p.workerId === worker.id && p.active) || null;
  const claims = DB.claims.filter(c => c.workerId === worker.id);
  res.json({ worker, policy, claims });
});

// POST activate policy
app.post("/api/policies/activate", (req, res) => {
  const { workerId } = req.body;
  const worker = DB.workers.find(w => w.id === workerId);
  if (!worker) return res.status(404).json({ error: "Worker not found" });

  const existing = DB.policies.find(p => p.workerId === workerId && p.active);
  if (existing) return res.status(409).json({ error: "Already active", policy: existing });

  const policy = {
    id: uid(), workerId,
    premium:     worker.premium,
    coverage:    1500,
    activatedAt: new Date().toISOString(),
    active:      true,
  };
  DB.policies.push(policy);
  res.status(201).json({ message: "Policy activated", policy });
});

// POST trigger claim — runs fraud check via ML before approving
app.post("/api/claims/trigger", async (req, res) => {
  const { workerId, eventType, eventValue, location_variance, speed_consistency,
          device_changes, ip_mismatch, similar_cluster } = req.body;

  const worker = DB.workers.find(w => w.id === workerId);
  if (!worker) return res.status(404).json({ error: "Worker not found" });

  const policy = DB.policies.find(p => p.workerId === workerId && p.active);
  if (!policy) return res.status(400).json({ error: "No active policy" });

  // Threshold check
  const thresholds = { rain: 40, heat: 42, aqi: 350, flood: 1 };
  if (Number(eventValue) < (thresholds[eventType] ?? Infinity))
    return res.json({ triggered: false, message: "Threshold not met" });

  // Calculate claim history for fraud context
  const workerClaims   = DB.claims.filter(c => c.workerId === workerId);
  const lastClaim      = workerClaims.at(-1);
  const hoursSinceLast = lastClaim
    ? (Date.now() - new Date(lastClaim.createdAt)) / 3600000
    : 999;
  const claimsThisWeek = workerClaims.filter(c =>
    Date.now() - new Date(c.createdAt) < 7 * 86400000
  ).length;
  const hoursSinceActivation =
    (Date.now() - new Date(policy.activatedAt)) / 3600000;

  // Run ML fraud detection
  const fraudResult = await callML("/ml/fraud", {
    worker_id:                workerId,
    claims_per_week:          claimsThisWeek,
    hours_since_activation:   hoursSinceActivation,
    location_variance:        location_variance   ?? 0.8,
    speed_consistency:        speed_consistency   ?? 0.75,
    device_changes_30d:       device_changes      ?? 0,
    ip_gps_mismatch:          ip_mismatch         ?? false,
    last_claim_interval_hours: hoursSinceLast,
    similar_device_cluster_size: similar_cluster  ?? 0,
  });

  // Decide based on trust score
  let claimStatus = "approved";
  let requiresReview = false;

  if (fraudResult) {
    if (fraudResult.action === "flag_and_review") {
      claimStatus    = "flagged";
      requiresReview = true;
    } else if (fraudResult.action === "quick_verification") {
      claimStatus = "pending_verification";
    }
  }

  const claim = {
    id: uid(), workerId, eventType,
    eventValue:    Number(eventValue),
    payout:        claimStatus === "approved" ? policy.coverage : 0,
    status:        claimStatus,
    fraudAnalysis: fraudResult,
    createdAt:     new Date().toISOString(),
  };

  DB.claims.push(claim);
  res.status(201).json({
    triggered: true,
    claim,
    fraud_check: fraudResult,
    ml_used:     !!fraudResult,
    message: claimStatus === "approved"
      ? `Claim approved. ₹${claim.payout} payout initiated.`
      : claimStatus === "flagged"
      ? "Claim flagged for manual review due to suspicious activity."
      : "Claim pending — quick verification required.",
  });
});

// GET stats
app.get("/api/stats", (_req, res) => {
  res.json({
    totalWorkers:    DB.workers.length,
    activePolicies:  DB.policies.filter(p => p.active).length,
    totalClaims:     DB.claims.length,
    approvedClaims:  DB.claims.filter(c => c.status === "approved").length,
    flaggedClaims:   DB.claims.filter(c => c.status === "flagged").length,
    totalPayout:     DB.claims.filter(c => c.status === "approved").reduce((s,c) => s + c.payout, 0),
  });
});

// ML health proxy
app.get("/api/ml/health", async (_req, res) => {
  try {
    const r = await fetch(`${ML_URL}/health`, { signal: AbortSignal.timeout(2000) });
    const d = await r.json();
    res.json({ ...d, ml_url: ML_URL });
  } catch {
    res.status(503).json({ status: "ml_service_down", ml_url: ML_URL });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`\nZeptoShield API    → http://localhost:${PORT}`);
  console.log(`ML Service expected→ ${ML_URL}`);
  console.log(`Weather API        → Open-Meteo (no key needed)\n`);
});
