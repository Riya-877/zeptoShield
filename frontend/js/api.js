// api.js — calls Node backend (port 5000) which calls ML service (8000) + Open-Meteo
// Falls back to rich mock if backend is offline

const BACKEND = "http://localhost:5000/api";

// ── CITY DATA for mock ──────────────────────────────────────────────────────
const CITY_RISK = {
  Mumbai:        { rain: 2400, temp: 38, aqi: 145, flood: 3.2, days: 42 },
  Chennai:       { rain: 1400, temp: 42, aqi: 120, flood: 2.1, days: 35 },
  Kolkata:       { rain: 1800, temp: 40, aqi: 180, flood: 2.8, days: 38 },
  Hyderabad:     { rain: 900,  temp: 42, aqi: 130, flood: 1.5, days: 28 },
  Bengaluru:     { rain: 970,  temp: 35, aqi: 95,  flood: 0.8, days: 22 },
  Delhi:         { rain: 750,  temp: 45, aqi: 280, flood: 0.5, days: 30 },
  Pune:          { rain: 720,  temp: 38, aqi: 85,  flood: 0.6, days: 18 },
  Ahmedabad:     { rain: 800,  temp: 46, aqi: 150, flood: 0.4, days: 24 },
  Jaipur:        { rain: 550,  temp: 46, aqi: 170, flood: 0.2, days: 15 },
  Surat:         { rain: 1200, temp: 38, aqi: 130, flood: 1.0, days: 20 },
  Lucknow:       { rain: 900,  temp: 44, aqi: 200, flood: 0.6, days: 22 },
  Bhubaneswar:   { rain: 1500, temp: 40, aqi: 100, flood: 1.8, days: 30 },
  Nagpur:        { rain: 1100, temp: 47, aqi: 110, flood: 0.7, days: 20 },
  Indore:        { rain: 950,  temp: 43, aqi: 120, flood: 0.5, days: 18 },
  Visakhapatnam: { rain: 1100, temp: 40, aqi: 90,  flood: 1.2, days: 25 },
  Coimbatore:    { rain: 700,  temp: 36, aqi: 80,  flood: 0.4, days: 12 },
};

const DEFAULT_RISK = { rain: 900, temp: 40, aqi: 120, flood: 0.8, days: 20 };

// ── MOCK ML ─────────────────────────────────────────────────────────────────
function mockPremium(city, weeklyEarnings = 6000) {
  const cf = CITY_RISK[city] || DEFAULT_RISK;
  const rainS  = Math.min(cf.rain  / 2400, 1.0);
  const tempS  = Math.max((cf.temp - 30) / 20, 0);
  const aqiS   = Math.min(cf.aqi   / 350,  1.0);
  const floodS = Math.min(cf.flood / 4,    1.0);
  const overall = rainS*0.30 + tempS*0.20 + aqiS*0.15 + floodS*0.35;
  const premium = Math.round(20 + overall * 30 + (weeklyEarnings / 12000) * 10);

  return {
    premium,
    risk_score: parseFloat(overall.toFixed(3)),
    confidence: 0.91,
    breakdown: {
      rainfall_risk:    parseFloat((rainS  * 100).toFixed(1)),
      temperature_risk: parseFloat((tempS  * 100).toFixed(1)),
      aqi_risk:         parseFloat((aqiS   * 100).toFixed(1)),
      flood_risk:       parseFloat((floodS * 100).toFixed(1)),
      overall_risk:     parseFloat((overall* 100).toFixed(1)),
    },
    city_stats: {
      avg_annual_rainfall_mm: cf.rain,
      max_recorded_temp_c:    cf.temp,
      avg_aqi:                cf.aqi,
      flood_events_per_year:  cf.flood,
      avg_disruption_days:    cf.days,
    },
    model: "GradientBoostingRegressor (mock)",
    features_used: 7,
  };
}

function mockFraud(workerData = {}) {
  const {
    claims_per_week = 0,
    hours_since_activation = 48,
    location_variance = 0.8,
    speed_consistency = 0.8,
    device_changes = 0,
    ip_mismatch = false,
    last_claim_interval = 48,
    cluster_size = 0,
  } = workerData;

  let fraudProb = 0.05;
  if (claims_per_week > 3)         fraudProb += 0.25;
  if (hours_since_activation < 2)  fraudProb += 0.15;
  if (location_variance < 0.3)     fraudProb += 0.20;
  if (speed_consistency < 0.3)     fraudProb += 0.15;
  if (ip_mismatch)                 fraudProb += 0.20;
  if (device_changes >= 3)         fraudProb += 0.10;
  if (cluster_size >= 3)           fraudProb += 0.15;
  if (last_claim_interval < 2)     fraudProb += 0.20;

  fraudProb = Math.min(fraudProb, 0.98);
  const trustScore = Math.round((1 - fraudProb) * 0.65 * 100 + 0.35 * 70);
  const score = Math.min(Math.max(trustScore, 0), 100);

  const level  = score >= 80 ? "Low"    : score >= 55 ? "Medium" : "High";
  const action = score >= 80 ? "instant_payout" : score >= 55 ? "quick_verification" : "flag_and_review";
  const color  = score >= 80 ? "green"  : score >= 55 ? "amber"  : "red";

  const signals = [];
  if (claims_per_week > 3)         signals.push({ flag: "High claim frequency",        weight: "high",   value: `${claims_per_week}/week` });
  if (hours_since_activation < 2)  signals.push({ flag: "Very new account",             weight: "medium", value: `${hours_since_activation.toFixed(1)}h old` });
  if (location_variance < 0.3)     signals.push({ flag: "Suspiciously static location", weight: "high",   value: `variance ${location_variance}` });
  if (ip_mismatch)                  signals.push({ flag: "IP/GPS mismatch detected",     weight: "high",   value: "confirmed" });
  if (device_changes >= 3)          signals.push({ flag: "Frequent device changes",      weight: "medium", value: `${device_changes} in 30d` });
  if (cluster_size >= 3)            signals.push({ flag: "Device cluster detected",      weight: "high",   value: `${cluster_size} similar devices` });

  return {
    trust_score: score, risk_level: level, action, color,
    fraud_probability: parseFloat((fraudProb * 100).toFixed(1)),
    anomaly_score: parseFloat((0.3 - fraudProb * 0.8).toFixed(3)),
    signals,
    models_used: ["RandomForestClassifier (mock)", "IsolationForest (mock)"],
    explanation: "We don't trust location alone — we trust behavior.",
  };
}

function mockWeather(city) {
  const cf = CITY_RISK[city] || DEFAULT_RISK;
  // Simulate plausible current readings near historical averages
  return {
    temperature_c:   parseFloat((cf.temp - 5 + Math.random() * 8).toFixed(1)),
    rainfall_mm:     parseFloat((Math.random() * 20).toFixed(1)),
    current_rain_mm: parseFloat((Math.random() * 5).toFixed(1)),
    aqi:             Math.round(cf.aqi * (0.7 + Math.random() * 0.6)),
    pm25:            Math.round(cf.aqi * 0.4 * Math.random()),
    source:          "mock (backend offline)",
    fetched_at:      new Date().toISOString(),
  };
}

// ── IN-MEMORY STORE (mock backend) ─────────────────────────────────────────
const MOCK_DB = { workers: [], policies: [], claims: [], _id: 100 };
const mockUid = () => MOCK_DB._id++;

function mockRiskTier(score) {
  return score >= 0.6 ? "high" : score >= 0.35 ? "medium" : "low";
}

const MOCK_ROUTES = {
  async "POST /workers/register"(body) {
    const { name, phone, city, weeklyEarnings } = body;
    const ex = MOCK_DB.workers.find(w => w.phone === phone);
    if (ex) return { error: "Already registered", worker: ex };
    const ml = mockPremium(city, weeklyEarnings);
    const w = {
      id: mockUid(), name, phone, city,
      weeklyEarnings: parseFloat(weeklyEarnings) || 6000,
      premium:    ml.premium,
      riskScore:  ml.risk_score,
      riskTier:   mockRiskTier(ml.risk_score),
      mlBreakdown: ml.breakdown,
      registeredAt: new Date().toISOString(),
    };
    MOCK_DB.workers.push(w);
    return { message: "Registered successfully", worker: w, ml_used: true };
  },

  async "GET /workers/:phone"({ phone }) {
    const w = MOCK_DB.workers.find(w => w.phone === phone);
    if (!w) return { error: "Worker not found" };
    const policy = MOCK_DB.policies.find(p => p.workerId === w.id && p.active) || null;
    const claims = MOCK_DB.claims.filter(c => c.workerId === w.id);
    return { worker: w, policy, claims };
  },

  async "POST /policies/activate"({ workerId }) {
    const w = MOCK_DB.workers.find(w => w.id === workerId);
    if (!w) return { error: "Worker not found" };
    const ex = MOCK_DB.policies.find(p => p.workerId === workerId && p.active);
    if (ex) return { error: "Already active", policy: ex };
    const p = { id: mockUid(), workerId, premium: w.premium, coverage: 1500, activatedAt: new Date().toISOString(), active: true };
    MOCK_DB.policies.push(p);
    return { message: "Policy activated", policy: p };
  },

  async "POST /claims/trigger"(body) {
    const { workerId, eventType, eventValue, ...fraudInputs } = body;
    const w = MOCK_DB.workers.find(w => w.id === workerId);
    if (!w) return { error: "Worker not found" };
    const policy = MOCK_DB.policies.find(p => p.workerId === workerId && p.active);
    if (!policy) return { error: "No active policy" };
    const thresholds = { rain: 40, heat: 42, aqi: 350, flood: 1 };
    if (Number(eventValue) < (thresholds[eventType] ?? Infinity))
      return { triggered: false, message: "Threshold not met" };

    const workerClaims = MOCK_DB.claims.filter(c => c.workerId === workerId);
    const lastClaim = workerClaims.at(-1);
    const hoursSinceLast = lastClaim ? (Date.now() - new Date(lastClaim.createdAt)) / 3600000 : 999;
    const claimsThisWeek = workerClaims.filter(c => Date.now() - new Date(c.createdAt) < 7*86400000).length;
    const hoursSinceActivation = (Date.now() - new Date(policy.activatedAt)) / 3600000;

    const fraudResult = mockFraud({
      claims_per_week:         claimsThisWeek,
      hours_since_activation:  hoursSinceActivation,
      location_variance:       fraudInputs.location_variance ?? 0.8,
      speed_consistency:       fraudInputs.speed_consistency ?? 0.75,
      device_changes:          fraudInputs.device_changes ?? 0,
      ip_mismatch:             fraudInputs.ip_mismatch ?? false,
      last_claim_interval:     hoursSinceLast,
      cluster_size:            fraudInputs.similar_cluster ?? 0,
    });

    const status = fraudResult.action === "flag_and_review"
      ? "flagged"
      : fraudResult.action === "quick_verification"
      ? "pending_verification"
      : "approved";

    const claim = {
      id: mockUid(), workerId, eventType, eventValue: Number(eventValue),
      payout: status === "approved" ? policy.coverage : 0,
      status, fraudAnalysis: fraudResult,
      createdAt: new Date().toISOString(),
    };
    MOCK_DB.claims.push(claim);
    return {
      triggered: true, claim, fraud_check: fraudResult, ml_used: true,
      message: status === "approved"
        ? `Claim approved. ₹${claim.payout} payout initiated.`
        : status === "flagged"
        ? "Claim flagged for manual review."
        : "Claim pending quick verification.",
    };
  },

  async "GET /weather/:city"({ city }) {
    const weather = mockWeather(city);
    const ml = mockPremium(city);
    return {
      city, weather,
      ml_risk_profile: {
        city, risk_score: ml.risk_score, risk_tier: mockRiskTier(ml.risk_score),
        live_alerts: [], any_triggered: false,
        risk_components: ml.breakdown,
      },
    };
  },
};

// ── PUBLIC API ───────────────────────────────────────────────────────────────
async function api(method, path, body = null) {
  try {
    const opts = { method, headers: { "Content-Type": "application/json" }, signal: AbortSignal.timeout(2000) };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(BACKEND + path, opts);
    return await res.json();
  } catch (_) { /* fall through to mock */ }

  // Mock routing
  const normPath = path.replace(/\/workers\/[^/]+/, "/workers/:phone")
                       .replace(/\/weather\/[^/]+/, "/weather/:city");
  const key = `${method} ${normPath}`;
  const handler = MOCK_ROUTES[key];
  if (!handler) return { error: "Unknown route" };

  const params = {};
  if (path.includes("/workers/") && method === "GET") params.phone = path.split("/workers/")[1];
  if (path.includes("/weather/"))  params.city  = path.split("/weather/")[1];

  return handler(body ? { ...body, ...params } : params);
}

window.ZS = { api, CITY_RISK, mockPremium, mockFraud };
