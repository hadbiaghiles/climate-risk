/**
 * Climate Risk AI - Dashboard JavaScript
 * Gestion des interactions, appels API, et graphiques
 */

// =============================================
// CONFIGURATION
// =============================================
const API_BASE = '';  // Same origin

// Chart.js default config for dark theme
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";

// =============================================
// INITIALIZATION
// =============================================

async function fetchFirst(urls) {
    for (const url of urls) {
        try {
            const res = await fetch(url);
            if (!res.ok) continue;
            const data = await res.json();
            if (data && !data.error) return data;
        } catch (_) {}
    }
    return null;
}

document.addEventListener('DOMContentLoaded', () => {
    checkModelStatus();
    loadModelMetrics();
    loadHistoryData();
    syncSliders();
    
    // Si on est sur la page analytics
    if (document.getElementById('chartConfusionMatrix')) {
        loadPerformanceData();
    }
});

/**
 * Synchronise les sliders avec les inputs numériques bidirectionnellement
 */
function syncSliders() {
    const inputs = document.querySelectorAll('.input-group');
    inputs.forEach(group => {
        const numberInput = group.querySelector('input[type="number"]');
        const rangeInput = group.querySelector('input[type="range"]');
        
        if (numberInput && rangeInput) {
            // Saisir un nombre met à jour le slider
            numberInput.addEventListener('input', () => {
                rangeInput.value = numberInput.value;
            });
            
            // Bouger le slider met à jour le nombre
            rangeInput.addEventListener('input', () => {
                numberInput.value = rangeInput.value;
            });
        }
    });
}


function riskLevel(p) {
    return p < 0.25 ? "LOW" : p < 0.5 ? "MEDIUM" : p < 0.75 ? "HIGH" : "CRITICAL";
}

function localPredict(d) {
    const t = Number(d.temperature_max ?? d.temperature ?? 25);
    const pr = Number(d.cumul_pluie_24h ?? d.precipitation ?? 0);
    const hu = Number(d.humidite_relative_air ?? d.humidity ?? 60);
    const ps = Number(d.pression_atmospherique ?? d.pressure ?? 1013);
    const ws = Number(d.vent_soutenu ?? d.wind_speed ?? 10);
    const sl = Number(d.anomalie_niveau_mer ?? d.sea_level ?? 0);
    let flood = Math.min((pr > 100 ? 0.9 : pr > 50 ? 0.7 : pr > 20 ? 0.4 : pr > 5 ? 0.2 : 0) + (sl > 30 ? 0.2 : sl > 15 ? 0.1 : 0), 1);
    let drought = Math.min((t > 40 ? 0.8 : t > 35 ? 0.6 : t > 30 ? 0.3 : 0) + (hu < 20 ? 0.3 : hu < 40 ? 0.15 : 0), 1);
    let hurr = Math.min((ws > 150 ? 0.95 : ws > 120 ? 0.8 : ws > 100 ? 0.6 : ws > 80 ? 0.4 : ws > 50 ? 0.2 : ws > 30 ? 0.1 : 0) + (ps < 950 ? 0.3 : ps < 980 ? 0.15 : ps < 1000 ? 0.05 : 0), 1);
    if (ws >= 118) hurr = Math.max(hurr, 0.99);
    if (pr >= 100) flood = Math.max(flood, 0.99);
    if (t >= 38 && pr < 5) drought = Math.max(drought, 0.99);
    let normal = Math.max(0, 1 - flood - drought - hurr);
    const payload = {
        flood_risk: flood, drought_risk: drought, hurricane_risk: hurr, normal_risk: normal,
        risk_level: { flood: riskLevel(flood), drought: riskLevel(drought), hurricane: riskLevel(hurr), normal: riskLevel(normal) },
        severity: { flood: riskLevel(flood), drought: riskLevel(drought), hurricane: riskLevel(hurr), normal: normal > 0.5 ? "LOW" : "MEDIUM" },
        confidence: Math.max(flood, drought, hurr, normal) * 100,
        inputs: d, timestamp: new Date().toISOString(), method: "demo_pages"
    };
    return payload;
}

async function fetchApi(url, opts) {
    const res = await fetch(url, opts);
    if (!res.ok) throw new Error("api " + res.status);
    return res.json();
}

// =============================================
// API CALLS
// =============================================

/**
 * Vérifie si le modèle est entraîné et prêt
 */
async function checkModelStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();

        const badge = document.getElementById('modelStatus');
        const dot = badge.querySelector('.status-dot');
        const text = badge.querySelector('.status-text');

        if (data.model_ready) {
            dot.classList.remove('not-ready');
            dot.classList.add('ready');
            text.textContent = 'Modèle prêt';
            document.getElementById('btnPredict').disabled = false;
            const details = document.getElementById('btnDetails');
            if (details) details.style.display = 'inline-flex';
        } else {
            dot.classList.remove('ready');
            dot.classList.add('not-ready');
            text.textContent = 'Non entraîné';
            document.getElementById('btnPredict').disabled = false; // allow to try
        }
    } catch (e) {
        console.error('Status check failed:', e);
        const badge = document.getElementById('modelStatus');
        if (badge) {
            const dot = badge.querySelector('.status-dot');
            const text = badge.querySelector('.status-text');
            if (dot) { dot.classList.remove('not-ready'); dot.classList.add('ready'); }
            if (text) text.textContent = 'Démo Pages';
        }
        const btn = document.getElementById('btnPredict');
        if (btn) btn.disabled = false;
        const details = document.getElementById('btnDetails');
        if (details) details.style.display = 'inline-flex';
    }
}

/**
 * Gère l'action du bouton intelligent (Charger ou Entraîner)
 */
async function handleLoadOrTrain() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();

        if (data.model_ready) {
            // Le modèle existe déjà, on le "charge" (déjà fait par l'API status et checkModelStatus)
            showToast('', 'Modèle chargé avec succès depuis le disque.');
            document.getElementById('btnDetails').style.display = 'inline-flex';
            checkModelStatus();
        } else {
            // Le modèle n'existe pas, on lance l'entraînement
            showToast('', 'Modèle non trouvé. Lancement de l\'entraînement automatique...');
            trainModel();
        }
    } catch (e) {
        showToast('', `Erreur de statut: ${e.message}`);
    }
}

/**
 * Lance l'entraînement du modèle
 */
async function trainModel() {
    const overlay = document.getElementById('trainingOverlay');
    const btnTrain = document.getElementById('btnTrain');

    overlay.classList.add('active');
    btnTrain.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/train`, { method: 'POST' });
        const data = await res.json();

        overlay.classList.remove('active');
        btnTrain.disabled = false;

        if (data.status === 'started' || data.status === 'success') {
            showToast('', data.message || 'Entraînement lancé.');
            for (let i = 0; i < 30; i++) {
                await new Promise(r => setTimeout(r, 4000));
                try {
                    const st = await fetchApi(`${API_BASE}/api/status`);
                    if (st.model_ready) break;
                } catch (_) { break; }
            }
            checkModelStatus();
            loadModelMetrics();
        } else {
            showToast('', `Erreur: ${data.error || 'Inconnue'}`);
        }
    } catch (e) {
        overlay.classList.remove('active');
        btnTrain.disabled = false;
        showToast('', `Erreur de connexion: ${e.message}`);
    }
}

/**
 * Effectue une prédiction de risque
 */
async function predict(event) {
    event.preventDefault();
    const btnPredict = document.getElementById('btnPredict');
    btnPredict.disabled = true;
    btnPredict.innerHTML = '<span class="btn-icon">⏳</span> Analyse en cours...';

    const payload = {
        temperature_max: parseFloat(document.getElementById('temperature_max').value),
        cumul_pluie_24h: parseFloat(document.getElementById('cumul_pluie_24h').value),
        humidite_relative_air: parseFloat(document.getElementById('humidite_relative_air').value),
        vent_soutenu: parseFloat(document.getElementById('vent_soutenu').value),
        pression_atmospherique: parseFloat(document.getElementById('pression_atmospherique').value),
        anomalie_niveau_mer: parseFloat(document.getElementById('anomalie_niveau_mer').value),
        jours_sans_pluie: parseFloat(document.getElementById('jours_sans_pluie').value),
        humidite_des_sols: parseFloat(document.getElementById('humidite_des_sols').value)
    };

    try {
        const res = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (data.error) {
            showToast('⚠️', data.error);
        } else {
            updateRiskCards(data);
            updateResultsPanel(data);
            showToast('', 'Analyse SOTA terminée !');
        }
    } catch (e) {
        const data = localPredict(payload);
        updateRiskCards(data);
        updateResultsPanel(data);
        showToast('', 'Analyse (mode démo Pages).');
    } finally {
        btnPredict.disabled = false;
        btnPredict.innerHTML = '<span class="btn-icon">🔮</span> Analyser les Risques';
    }
}
/**
 * Charge les métriques détaillées du modèle
 */
async function loadModelMetrics() {
    try {
        const data = await fetchFirst([
            `${API_BASE}/api/model-info`,
            'data/evaluation.json',
            'static/data/evaluation.json'
        ]);

        if (data && data.accuracy) {
            const accEl = document.getElementById('metricAccuracy');
            const lossEl = document.getElementById('metricLoss');
            const timeEl = document.getElementById('metricTimestamp');

            if (accEl) accEl.textContent = `${(data.accuracy * 100).toFixed(2)}%`;
            if (lossEl) lossEl.textContent = data.loss.toFixed(4);
            if (timeEl && data.timestamp) timeEl.textContent = `Dernière évaluation: ${data.timestamp}`;
        }
    } catch (e) {
        console.error('Failed to load metrics:', e);
    }
}

function updateRiskMetric(id, accuracy) {
    const textEl = document.getElementById(`riskAcc${id}`);
    const fillEl = document.getElementById(`riskFill${id}`);
    
    if (textEl && fillEl) {
        textEl.textContent = `${(accuracy * 100).toFixed(1)}%`;
        fillEl.style.width = `${accuracy * 100}%`;
        fillEl.style.background = getRiskColor(accuracy * 100);
    }
}

// =============================================
// UI UPDATES
// =============================================

/**
 * Met à jour les cartes de risque avec les résultats de prédiction
 */
function updateRiskCards(data) {
    // Flood
    updateGauge('Flood', data.flood_risk, data.risk_level.flood);
    // Drought
    updateGauge('Drought', data.drought_risk, data.risk_level.drought);
    // Hurricane
    updateGauge('Hurricane', data.hurricane_risk, data.risk_level.hurricane);
}

/**
 * Met à jour un gauge circulaire
 */
function updateGauge(type, value, level) {
    const circumference = 326.7;
    const offset = circumference - (value * 100 / 100) * circumference;

    const gaugeFill = document.getElementById(`gaugeFill${type}`);
    const gaugeValue = document.getElementById(`value${type}`);
    const riskLevel = document.getElementById(`level${type}`);

    gaugeFill.style.strokeDashoffset = offset;

    // Special handling for Normal weather - never red
    let color;
    if (type === 'Normal') {
        color = value > 0.5 ? '#10b981' : '#3b82f6'; // Green if dominant, blue otherwise
        // Replace danger levels with positive ones
        if (level === 'HIGH' || level === 'CRITICAL') {
            level = value > 0.7 ? 'DOMINANT' : 'PROBABLE';
        }
    } else {
        color = getRiskColor(value * 100);
    }
    gaugeFill.style.stroke = color;

    // Format percentage with 2 decimals
    gaugeValue.textContent = `${(value * 100).toFixed(2)}%`;
    gaugeValue.style.color = color;

    // Update risk level text if element exists (center of card)
    if (riskLevel) {
        riskLevel.textContent = level;
        riskLevel.style.color = color;
        riskLevel.style.background = hexToRgba(color, 0.15);
    }
}

/**
 * Met à jour le panneau de résultats détaillés avec SOTA indicators
 */
function updateResultsPanel(data) {
    document.getElementById('resultsPlaceholder').style.display = 'none';
    document.getElementById('resultsContent').style.display = 'block';
    
    // Confidence - formatted with 2 decimals
    const confBadge = document.getElementById('confidenceBadge');
    confBadge.style.display = 'flex';
    document.getElementById('confValue').textContent = `${(data.confidence).toFixed(2)}%`;

    // Indicators (Flood, Drought, Hurricane, Normal)
    updateIndicator('Flood', data.risk_level.flood, data.severity.flood);
    updateIndicator('Drought', data.risk_level.drought, data.severity.drought);
    updateIndicator('Hurricane', data.risk_level.hurricane, data.severity.hurricane);
    updateIndicator('Normal', data.risk_level.normal, data.severity.normal);

    // Raw Bars
    animateBar('Flood', data.flood_risk);
    animateBar('Drought', data.drought_risk);
    animateBar('Hurricane', data.hurricane_risk);
    animateBar('Normal', data.normal_risk);

    // Insights
    document.getElementById('resTimestamp').textContent = data.timestamp;
    document.getElementById('resInputs').textContent = JSON.stringify(data.inputs).replace(/[{}"]/g, '').replace(/:/g, ': ').replace(/,/g, ' | ');
}

function updateIndicator(id, level, severity) {
    const el = document.getElementById(`indicator${id}`);
    const lbl = document.getElementById(`levelLabel${id}`);
    const desc = document.getElementById(`desc${id}`);
    
    // Special handling for Normal weather
    if (id === 'Normal') {
        // Never show danger colors or HIGH/CRITICAL
        if (level === 'HIGH' || level === 'CRITICAL') {
            level = 'DOMINANT';
        }
        lbl.style.color = '#10b981'; // Always green
    }
    
    lbl.textContent = level;
    desc.textContent = severity;
    
    // Remove old classes
    el.classList.remove('low', 'medium', 'high');
    lbl.classList.remove('LOW', 'MEDIUM', 'HIGH');
    
    // Add new classes
    el.classList.add(level.toLowerCase());
    lbl.classList.add(level);
}

function animateBar(type, value) {
    const bar = document.getElementById(`result${type}Bar`);
    const pct = document.getElementById(`result${type}Pct`);
    const percentage = (value * 100).toFixed(2);
    let color = getRiskColor(percentage);

    if (type === 'Normal') {
        color = '#10b981'; // green for normal
    }

    setTimeout(() => {
        bar.style.width = `${percentage}%`;
    }, 100);

    pct.textContent = `${percentage}%`;
    pct.style.color = color;
}

// =============================================
// PERFORMANCE ANALYTICS (DETAILS PAGE)
// =============================================

async function loadPerformanceData() {
    try {
        const data = await fetchFirst([
            `${API_BASE}/api/performance-details`,
            'data/evaluation.json',
            'static/data/evaluation.json'
        ]);
        
        if (!data || (!data.confusion_matrix || data.confusion_matrix.length === 0)) {
            console.warn("Analytics data not available yet.");

            // Display placeholder notices for the charts
            const setPlaceholder = (canvasId, message) => {
                const canvas = document.getElementById(canvasId);
                if (canvas && canvas.parentElement) {
                    canvas.parentElement.innerHTML = '<div style="height: 100%; display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-weight:500;">'+ message +'</div>';
                }
            };

            setPlaceholder('chartConfusionMatrix', 'Aucune matrice de confusion disponible pour le moment. Entraînez le modèle et rechargez.');
            setPlaceholder('chartROC', 'Aucune courbe ROC disponible pour le moment. Entraînez le modèle et rechargez.');
            setPlaceholder('chartFeatureImportance', 'Aucune importance des features disponible pour le moment. Entraînez le modèle et rechargez.');
            return;
        }

        renderHeatmap(data.confusion_matrix, data.classes);
        renderROC(data.roc_curves, data.classes);
        renderImportance(data.feature_importance);
        
        if (data.timestamp) {
            const timeEl = document.getElementById('metricTimestamp');
            if (timeEl) timeEl.textContent = `Dernière évaluation: ${data.timestamp}`;
        }
    } catch (e) {
        console.error("Error loading performance data:", e);
    }
}

function renderHeatmap(matrix, classes) {
    const ctx = document.getElementById('chartConfusionMatrix').getContext('2d');
    
    new Chart(ctx, {
        type: 'matrix',
        data: {
            datasets: [{
                label: 'Confusion Matrix',
                data: matrix,
                backgroundColor(context) {
                    const value = context.dataset.data[context.dataIndex].v;
                    const maxV = Math.max(...context.dataset.data.map(d => d.v), 1);
                    const alpha = 0.15 + 0.85 * (value / maxV);
                    return `rgba(59, 130, 246, ${alpha})`;
                },
                borderWidth: 1,
                borderColor: 'rgba(255,255,255,0.1)',
                width: ({chart}) => (chart.chartArea || {}).width / classes.length - 1,
                height: ({chart}) => (chart.chartArea || {}).height / classes.length - 1
            }]
        },
        options: {
            maintainAspectRatio: false,
            plugins: {
                legend: false,
                tooltip: {
                    callbacks: {
                        title() { return ''; },
                        label(context) {
                            const v = context.dataset.data[context.dataIndex];
                            return [`Réalité: ${v.y}`, `Prédiction: ${v.x}`, `Nombre: ${v.v}`];
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'category',
                    labels: classes,
                    title: { display: true, text: 'Prédiction' },
                    grid: { display: false }
                },
                y: {
                    type: 'category',
                    labels: classes,
                    title: { display: true, text: 'Réalité' },
                    grid: { display: false }
                }
            }
        }
    });
}

function renderROC(curves, classes) {
    const ctx = document.getElementById('chartROC').getContext('2d');
    const datasets = classes.map((cls, i) => {
        const colors = ['#3b82f6', '#f59e0b', '#8b5cf6'];
        return {
            label: `${cls} (AUC: ${curves[cls].auc.toFixed(2)})`,
            data: curves[cls].fpr.map((f, idx) => ({x: f, y: curves[cls].tpr[idx]})),
            borderColor: colors[i],
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
            tension: 0.1
        };
    });

    // Add baseline
    datasets.push({
        label: 'Base (Chance)',
        data: [{x:0, y:0}, {x:1, y:1}],
        borderColor: 'rgba(255,255,255,0.2)',
        borderDash: [5, 5],
        borderWidth: 1,
        pointRadius: 0,
        fill: false
    });

    new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            maintainAspectRatio: false,
            scales: {
                x: { type: 'linear', title: { display: true, text: 'Taux Faux Positifs' }, min: 0, max: 1 },
                y: { type: 'linear', title: { display: true, text: 'Taux Vrais Positifs' }, min: 0, max: 1 }
            },
            plugins: {
                legend: { position: 'bottom', labels: { boxWidth: 12, padding: 20 } }
            }
        }
    });
}

function renderImportance(importance) {
    const ctx = document.getElementById('chartFeatureImportance').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: importance.map(i => i.feature),
            datasets: [{
                label: 'Baisse d\'Accuracy',
                data: importance.map(i => i.importance),
                backgroundColor: 'rgba(6, 182, 212, 0.6)',
                borderColor: '#06b6d4',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            maintainAspectRatio: false,
            plugins: {
                legend: false
            },
            scales: {
                x: { title: { display: true, text: 'Importance (Permutation)' } }
            }
        }
    });
}

// =============================================
// CHARTS
// =============================================

let chartTempPrecip = null;
let chartRisks = null;
let chartWindPressure = null;
let chartHumiditySea = null;

/**
 * Charge et affiche les données historiques dans les graphiques
 */
async function loadHistoryData() {
    try {
        const data = await fetchFirst([
            `${API_BASE}/api/history`,
            'data/history.json',
            'static/data/history.json'
        ]);
        if (!data) {
            const infoEl = document.getElementById('historyInfo');
            if (infoEl) infoEl.textContent = 'Données historiques indisponibles';
            return;
        }

        // Simplifier les labels (1 sur 7 pour lisibilité)
        const labels = data.dates.filter((_, i) => i % 7 === 0);
        const sampleEvery = (arr) => arr.filter((_, i) => i % 7 === 0);

        createTempPrecipChart(labels, sampleEvery(data.temperature), sampleEvery(data.precipitation));
        createRisksChart(labels, sampleEvery(data.flood_risk), sampleEvery(data.drought_risk), sampleEvery(data.hurricane_risk), sampleEvery(data.normal_risk));
        
        // Fix: Ensure these fields exist before trying to sample/create charts
        if (data.wind_speed && data.pressure) {
            createWindPressureChart(labels, sampleEvery(data.wind_speed), sampleEvery(data.pressure));
        }
        if (data.humidity && data.sea_level) {
            createHumiditySeaChart(labels, sampleEvery(data.humidity), sampleEvery(data.sea_level));
        }

        // Add info to UI if elements exist
        const infoEl = document.getElementById('historyInfo');
        if (infoEl) {
            const start = data.info && data.info.start_date;
            const end = data.info && data.info.end_date;
            infoEl.textContent = start && end
                ? `Période: ${start} à ${end}`
                : `Échantillon: ${data.dates.length} points`;
        }

    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

function createTempPrecipChart(labels, temp, precip) {
    const ctx = document.getElementById('chartTempPrecip').getContext('2d');
    if (chartTempPrecip) chartTempPrecip.destroy();

    chartTempPrecip = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Température (°C)',
                    data: temp,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 2,
                    yAxisID: 'y'
                },
                {
                    label: 'Précipitations (mm)',
                    data: precip,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 2,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            interaction: { intersect: false, mode: 'index' },
            plugins: { legend: { position: 'top', labels: { usePointStyle: true, pointStyle: 'circle', padding: 16 } } },
            scales: {
                x: { display: true, ticks: { maxTicksLimit: 12, maxRotation: 45 } },
                y: { position: 'left', title: { display: true, text: '°C' } },
                y1: { position: 'right', title: { display: true, text: 'mm' }, grid: { drawOnChartArea: false } }
            }
        }
    });
}

function createRisksChart(labels, flood, drought, hurricane, normal) {
    const ctx = document.getElementById('chartRisks').getContext('2d');
    if (chartRisks) chartRisks.destroy();

    // Compute rolling averages for smooth visualization
    const rolling = (arr, window = 4) => {
        return arr.map((_, i) => {
            const start = Math.max(0, i - window + 1);
            const slice = arr.slice(start, i + 1);
            return slice.reduce((a, b) => a + b, 0) / slice.length;
        });
    };

    chartRisks = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: '🌊 Inondation',
                    data: rolling(flood),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.15)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 2
                },
                {
                    label: '☀️ Sécheresse',
                    data: rolling(drought),
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.15)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 2
                },
                {
                    label: '🌀 Ouragan',
                    data: rolling(hurricane),
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.15)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 2
                },
                {
                    label: '☀️ Temps Normal',
                    data: rolling(normal),
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.15)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            interaction: { intersect: false, mode: 'index' },
            plugins: { legend: { position: 'top', labels: { usePointStyle: true, pointStyle: 'circle', padding: 16 } } },
            scales: {
                x: { ticks: { maxTicksLimit: 12, maxRotation: 45 } },
                y: { min: 0, max: 1, title: { display: true, text: 'Probabilité' } }
            }
        }
    });
}

function createWindPressureChart(labels, wind, pressure) {
    const ctx = document.getElementById('chartWindPressure').getContext('2d');
    if (chartWindPressure) chartWindPressure.destroy();

    chartWindPressure = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Vent (km/h)',
                    data: wind,
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 2,
                    yAxisID: 'y'
                },
                {
                    label: 'Pression (hPa)',
                    data: pressure,
                    borderColor: '#ec4899',
                    backgroundColor: 'rgba(236, 72, 153, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 2,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            interaction: { intersect: false, mode: 'index' },
            plugins: { legend: { position: 'top', labels: { usePointStyle: true, pointStyle: 'circle', padding: 16 } } },
            scales: {
                x: { ticks: { maxTicksLimit: 12, maxRotation: 45 } },
                y: { position: 'left', title: { display: true, text: 'km/h' } },
                y1: { position: 'right', title: { display: true, text: 'hPa' }, grid: { drawOnChartArea: false } }
            }
        }
    });
}

function createHumiditySeaChart(labels, humidity, seaLevel) {
    const ctx = document.getElementById('chartHumiditySea').getContext('2d');
    if (chartHumiditySea) chartHumiditySea.destroy();

    chartHumiditySea = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Humidité (%)',
                    data: humidity,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 2,
                    yAxisID: 'y'
                },
                {
                    label: 'Niveau mer (cm)',
                    data: seaLevel,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 2,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            interaction: { intersect: false, mode: 'index' },
            plugins: { legend: { position: 'top', labels: { usePointStyle: true, pointStyle: 'circle', padding: 16 } } },
            scales: {
                x: { ticks: { maxTicksLimit: 12, maxRotation: 45 } },
                y: { position: 'left', title: { display: true, text: '%' } },
                y1: { position: 'right', title: { display: true, text: 'cm' }, grid: { drawOnChartArea: false } }
            }
        }
    });
}

// =============================================
// UTILITY FUNCTIONS
// =============================================

function getRiskColor(value) {
    if (value < 20) return '#10b981';       // emerald
    if (value < 40) return '#22c55e';       // green
    if (value < 60) return '#f59e0b';       // amber
    if (value < 80) return '#ef4444';       // red
    return '#dc2626';                        // dark red
}

function hexToRgba(hex, alpha) {
    // Si ce n'est pas un hex (déjà en rgba ou nom de couleur), on le retourne tel quel
    if (!hex || typeof hex !== 'string' || !hex.startsWith('#')) {
        return hex;
    }
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function showToast(icon, message) {
    const toast = document.getElementById('toast');
    const toastIcon = document.getElementById('toastIcon');
    const toastMessage = document.getElementById('toastMessage');

    toastIcon.textContent = icon;
    toastMessage.textContent = message;

    toast.classList.add('show');

    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}
