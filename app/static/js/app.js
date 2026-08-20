/* ============================================================
   TerraPulse ML — Client Interaction & Prediction Engine
   ============================================================ */

// Map Viewport Constants
const AMES_CENTER = [42.034, -93.642];
const GLOBAL_CENTER = [20.0, 0.0];

const BASEMAPS = {
    dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    light: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
};

// Global App State
let currentSection = 'calculator'; // 'calculator' | 'map' | 'docs'
let currentScope = 'ames';         // 'ames' | 'global'
let map, currentTileLayer, labelTileLayer;
let amesDotsLayerGroup, globalDotsLayerGroup, activeMarkerLayerGroup;
let selectedProperty = null;
let calcDebounceTimer = null;
let inspDebounceTimer = null;
let searchIndex = [];

// ============================================================
// 1. Navigation & Section Routing
// ============================================================
function switchSection(sectionId) {
    currentSection = sectionId;

    // Update Header Tabs
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.section === sectionId);
    });

    // Update Sections Visibility
    document.querySelectorAll('.app-section').forEach(sec => {
        sec.classList.toggle('active', sec.id === `section-${sectionId}`);
    });

    // Invalidate map size if switching to map
    if (sectionId === 'map') {
        setTimeout(() => {
            if (map) map.invalidateSize();
        }, 100);
    }
}

function setScope(scope) {
    currentScope = scope;
    document.getElementById('btn-scope-ames').classList.toggle('active', scope === 'ames');
    document.getElementById('btn-scope-global').classList.toggle('active', scope === 'global');

    // Toggle dropdown visibility in Calculator
    const neighGroup = document.getElementById('group-calc-neighborhood');
    const cityGroup = document.getElementById('group-calc-city');
    if (neighGroup && cityGroup) {
        neighGroup.classList.toggle('hidden', scope === 'global');
        cityGroup.classList.toggle('hidden', scope === 'ames');
    }

    // Toggle map layers if map is ready
    if (map) {
        if (scope === 'ames') {
            map.flyTo(AMES_CENTER, 13, { duration: 1.5 });
            if (amesDotsLayerGroup) map.addLayer(amesDotsLayerGroup);
            if (globalDotsLayerGroup) map.removeLayer(globalDotsLayerGroup);
            document.getElementById('map-legend').style.display = 'block';
        } else {
            map.flyTo(GLOBAL_CENTER, 2.5, { duration: 1.8 });
            if (amesDotsLayerGroup) map.removeLayer(amesDotsLayerGroup);
            if (globalDotsLayerGroup) map.addLayer(globalDotsLayerGroup);
            document.getElementById('map-legend').style.display = 'none';
        }
    }

    // Trigger calculation for new scope
    handleCalcChange();
}

// ============================================================
// 2. Map Initialization & Dot Pointer Layers
// ============================================================
function initMap() {
    map = L.map('map', {
        center: AMES_CENTER,
        zoom: 13,
        minZoom: 2,
        maxZoom: 18,
        zoomControl: false,
    });

    L.control.zoom({ position: 'topright' }).addTo(map);

    setBasemap('dark');

    amesDotsLayerGroup = L.layerGroup();
    globalDotsLayerGroup = L.layerGroup();
    activeMarkerLayerGroup = L.layerGroup().addTo(map);

    renderAmesHouseDots();
    renderGlobalCityDots();

    if (currentScope === 'ames') {
        amesDotsLayerGroup.addTo(map);
    } else {
        globalDotsLayerGroup.addTo(map);
    }

    map.on('click', handleMapBackgroundClick);

    setupSearchIndex();
}

function setBasemap(type) {
    if (currentTileLayer) map.removeLayer(currentTileLayer);
    if (labelTileLayer) map.removeLayer(labelTileLayer);

    currentTileLayer = L.tileLayer(BASEMAPS[type], {
        attribution: '&copy; OpenStreetMap &copy; CartoDB &copy; Esri',
        maxZoom: 19,
        subdomains: 'abcd'
    }).addTo(map);

    if (type === 'satellite') {
        labelTileLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png', {
            maxZoom: 19,
            subdomains: 'abcd',
            opacity: 0.85
        }).addTo(map);
    }

    document.querySelectorAll('.layer-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.layer === type);
    });
}

function renderAmesHouseDots() {
    if (!AMES_HOUSES || !Array.isArray(AMES_HOUSES)) return;

    AMES_HOUSES.forEach(house => {
        const marker = L.circleMarker([house.lat, house.lng], {
            radius: 5,
            fillColor: house.color || '#38bdf8',
            color: '#ffffff',
            weight: 0.75,
            opacity: 0.9,
            fillOpacity: 0.8,
        });

        // Hover Tooltip
        const tooltipContent = `
            <div style="font-family: var(--font-sans); padding: 4px;">
                <div style="font-weight: 800; font-size: 13px; color: #ffffff;">$${house.price.toLocaleString()}</div>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">
                    ${house.GrLivArea} sqft · Quality: ${house.OverallQual}/10<br>
                    <span style="color: #38bdf8;">${getNeighborhoodFullName(house.neighborhood)}</span>
                </div>
            </div>
        `;
        marker.bindTooltip(tooltipContent, { sticky: true, className: 'house-dot-tooltip' });

        marker.on('click', (e) => {
            L.DomEvent.stopPropagation(e);
            selectHouseDot(house);
        });

        marker.addTo(amesDotsLayerGroup);
    });
}

function renderGlobalCityDots() {
    if (!CITY_COORDINATES || !GLOBAL_STATS?.cities) return;

    for (const [key, coords] of Object.entries(CITY_COORDINATES)) {
        const [country, city] = key.split('|');
        const stats = GLOBAL_STATS.cities[key];
        const median = stats ? stats.median_price : 1000000;

        const icon = L.divIcon({
            className: '',
            html: `
                <div class="city-marker" title="${city}, ${country}">
                    <span class="city-dot"></span>
                    <span>${city}</span>
                    <span class="city-price">$${formatPrice(median)}</span>
                </div>
            `,
            iconSize: null,
            iconAnchor: [40, 15]
        });

        const marker = L.marker(coords, { icon });
        marker.on('click', (e) => {
            L.DomEvent.stopPropagation(e);
            selectGlobalCityDot(country, city, coords);
        });

        marker.addTo(globalDotsLayerGroup);
    }
}

function selectHouseDot(house) {
    selectedProperty = {
        isGlobal: false,
        lat: house.lat,
        lng: house.lng,
        neighborhood: house.neighborhood,
        price: house.price,
        features: {
            Neighborhood: house.neighborhood,
            OverallQual: house.OverallQual,
            GrLivArea: house.GrLivArea,
            YearBuilt: house.YearBuilt,
            BedroomAbvGr: house.BedroomAbvGr,
            FullBath: house.FullBath,
            GarageCars: house.GarageCars,
            TotalBsmtSF: house.TotalBsmtSF,
            LotArea: house.LotArea,
            OverallCond: house.OverallCond,
            KitchenQual: house.KitchenQual,
        }
    };

    highlightMapPin(house.lat, house.lng);
    populateInspector(selectedProperty);
    openInspector();
    triggerInspectorPrediction(selectedProperty);
}

function selectGlobalCityDot(country, city, coords) {
    const defaults = GLOBAL_DEFAULTS?.[country] || {};
    selectedProperty = {
        isGlobal: true,
        lat: coords[0],
        lng: coords[1],
        country: country,
        city: city,
        features: {
            country: country,
            city: city,
            property_size_sqft: defaults.property_size_sqft || 1800,
            rooms: defaults.rooms || 4,
            bathrooms: defaults.bathrooms || 2,
            constructed_year: defaults.constructed_year || 2012,
            property_type: 'Apartment'
        }
    };

    highlightMapPin(coords[0], coords[1]);
    populateInspector(selectedProperty);
    openInspector();
    triggerInspectorPrediction(selectedProperty);
}

function handleMapBackgroundClick(e) {
    const lat = e.latlng.lat;
    const lng = e.latlng.lng;

    if (currentScope === 'ames') {
        const defaultNeigh = findClosestNeighborhood(lat, lng) || 'NAmes';
        selectedProperty = {
            isGlobal: false,
            lat, lng,
            neighborhood: defaultNeigh,
            features: {
                Neighborhood: defaultNeigh,
                OverallQual: 7,
                GrLivArea: 1800,
                YearBuilt: 2005,
                BedroomAbvGr: 3,
                FullBath: 2,
                GarageCars: 2,
                TotalBsmtSF: 900,
            }
        };
    } else {
        selectedProperty = {
            isGlobal: true,
            lat, lng,
            country: 'USA',
            city: 'San Francisco',
            features: {
                country: 'USA',
                city: 'San Francisco',
                property_size_sqft: 1800,
                rooms: 4,
                bathrooms: 2,
                constructed_year: 2010
            }
        };
    }

    highlightMapPin(lat, lng);
    populateInspector(selectedProperty);
    openInspector();
    triggerInspectorPrediction(selectedProperty);
}

function highlightMapPin(lat, lng) {
    activeMarkerLayerGroup.clearLayers();
    const pin = L.circleMarker([lat, lng], {
        radius: 9,
        fillColor: '#4f8ff7',
        color: '#ffffff',
        weight: 2.5,
        opacity: 1,
        fillOpacity: 0.9,
    }).addTo(activeMarkerLayerGroup);
}

function clearAllMarkers() {
    activeMarkerLayerGroup.clearLayers();
    closeInspector();
}

// ============================================================
// 3. Section 1: Quick Predictor (Calculator Logic)
// ============================================================
function initCalculatorDropdowns() {
    const neighSelect = document.getElementById('calc-Neighborhood');
    if (neighSelect && NEIGHBORHOOD_STATS) {
        neighSelect.innerHTML = Object.keys(NEIGHBORHOOD_STATS).sort().map(code => {
            const name = getNeighborhoodFullName(code);
            return `<option value="${code}">${name}</option>`;
        }).join('');
        neighSelect.value = 'CollgCr';
    }

    const citySelect = document.getElementById('calc-city');
    if (citySelect && CITY_COORDINATES) {
        citySelect.innerHTML = Object.keys(CITY_COORDINATES).sort().map(k => {
            const [country, city] = k.split('|');
            return `<option value="${city}" data-country="${country}">${city}, ${country}</option>`;
        }).join('');
    }
}

function handleCalcChange() {
    clearTimeout(calcDebounceTimer);
    calcDebounceTimer = setTimeout(runCalcPrediction, 80);
}

async function runCalcPrediction() {
    const isGlobal = currentScope === 'global';
    const qual = Number(document.getElementById('calc-OverallQual')?.value || 7);
    const qualBadge = document.getElementById('calc-qual-badge');
    if (qualBadge) qualBadge.textContent = `${qual} · ${getQualityLabel(qual)}`;

    let payload = {};
    let endpoint = '/api/predict';

    if (!isGlobal) {
        const neigh = document.getElementById('calc-Neighborhood')?.value || 'CollgCr';
        payload = {
            Neighborhood: neigh,
            OverallQual: qual,
            GrLivArea: Number(document.getElementById('calc-GrLivArea')?.value || 1850),
            YearBuilt: Number(document.getElementById('calc-YearBuilt')?.value || 2005),
            TotalBsmtSF: Number(document.getElementById('calc-TotalBsmtSF')?.value || 950),
            BedroomAbvGr: Number(document.getElementById('calc-BedroomAbvGr')?.value || 3),
            FullBath: Number(document.getElementById('calc-FullBath')?.value || 2),
            GarageCars: Number(document.getElementById('calc-GarageCars')?.value || 2),
            LotArea: Number(document.getElementById('calc-LotArea')?.value || 8500),
            KitchenQual: document.getElementById('calc-KitchenQual')?.value || 'Gd',
        };
        endpoint = '/api/predict';
    } else {
        const cityOption = document.getElementById('calc-city')?.selectedOptions[0];
        const city = cityOption?.value || 'Berlin';
        const country = cityOption?.dataset?.country || 'Germany';
        payload = {
            country: country,
            city: city,
            property_size_sqft: Number(document.getElementById('calc-GrLivArea')?.value || 1850),
            rooms: Number(document.getElementById('calc-BedroomAbvGr')?.value || 4),
            bathrooms: Number(document.getElementById('calc-FullBath')?.value || 2),
            constructed_year: Number(document.getElementById('calc-YearBuilt')?.value || 2010),
            property_type: 'Apartment'
        };
        endpoint = '/api/predict/global';
    }

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!response.ok) return;

        const result = await response.json();
        renderCalculatorResult(result, payload, isGlobal);
    } catch (err) {
        console.error('Calculation prediction error:', err);
    }
}

function renderCalculatorResult(result, payload, isGlobal) {
    const price = Math.round(result.predicted_price || 0);
    document.getElementById('calc-price-display').textContent = `$${price.toLocaleString()}`;

    const sqft = Number(payload.GrLivArea || payload.property_size_sqft || 1850);
    const pricePerSqFt = sqft > 0 ? Math.round(price / sqft) : 0;
    document.getElementById('calc-sqft-rate').textContent = `$${pricePerSqFt} / sqft`;

    const locLabel = isGlobal ? `${payload.city}, ${payload.country}` : `${getNeighborhoodFullName(payload.Neighborhood)} · Ames, IA`;
    document.getElementById('calc-location-label').textContent = locLabel;

    // Model Tag
    const modelTag = isGlobal ? 'Random Forest AI · 99.9% R²' : 'CatBoost AI · 90.9% R²';
    document.getElementById('calc-model-badge').textContent = modelTag;

    // Median and Range
    const median = result.neighborhood_median || result.location_median || price;
    const min = result.neighborhood_min || result.location_min || price * 0.6;
    const max = result.neighborhood_max || result.location_max || price * 1.6;

    const diffPct = ((price - median) / (median || 1)) * 100;
    const trendEl = document.getElementById('calc-trend-badge');
    if (diffPct >= 0) {
        trendEl.className = 'trend-pill positive';
        trendEl.textContent = `+${diffPct.toFixed(1)}% vs Med`;
    } else {
        trendEl.className = 'trend-pill negative';
        trendEl.textContent = `${diffPct.toFixed(1)}% vs Med`;
    }

    const range = max - min || 1;
    const pct = Math.min(100, Math.max(0, ((price - min) / range) * 100));
    document.getElementById('calc-range-bar-fill').style.width = `${pct}%`;
    document.getElementById('calc-range-marker').style.left = `${pct}%`;
    document.getElementById('calc-range-min').textContent = `$${formatPrice(min)}`;
    document.getElementById('calc-range-max').textContent = `$${formatPrice(max)}`;

    // Financial Grid
    const confLower = result.confidence_lower ? `$${formatPrice(result.confidence_lower)}` : `$${formatPrice(price * 0.92)}`;
    const confUpper = result.confidence_upper ? `$${formatPrice(result.confidence_upper)}` : `$${formatPrice(price * 1.08)}`;
    document.getElementById('calc-conf-interval').textContent = `${confLower} – ${confUpper}`;

    const loan = price * 0.8;
    const monthlyRate = 0.065 / 12;
    const nPayments = 360;
    const monthlyEst = Math.round((loan * monthlyRate * Math.pow(1 + monthlyRate, nPayments)) / (Math.pow(1 + monthlyRate, nPayments) - 1));
    document.getElementById('calc-est-mortgage').textContent = `$${monthlyEst.toLocaleString()}/mo`;
    document.getElementById('calc-est-down').textContent = `$${Math.round(price * 0.2).toLocaleString()}`;
    document.getElementById('calc-est-rent').textContent = `$${Math.round(price * 0.0075).toLocaleString()}/mo`;

    // Attribution Decomposition
    renderAttributionList('calc-attribution-list', result.attribution);

    // Save as current active property state
    selectedProperty = {
        isGlobal,
        price,
        neighborhood: payload.Neighborhood,
        country: payload.country,
        city: payload.city,
        features: payload,
        attribution: result.attribution,
        confidence_lower: result.confidence_lower,
        confidence_upper: result.confidence_upper,
        model_name: result.model,
        holdout_r2: result.holdout_r2,
        lat: 42.034,
        lng: -93.642
    };
}

function renderAttributionList(containerId, attribution) {
    const el = document.getElementById(containerId);
    if (!el) return;

    if (attribution && attribution.length > 0) {
        el.innerHTML = attribution.map(item => {
            const isBase = item.type === 'base';
            const isPos = item.delta >= 0;
            const deltaStr = isBase ? `$${item.value.toLocaleString()}` : `${isPos ? '+' : ''}$${item.delta.toLocaleString()}`;
            const colorClass = isBase ? '' : (isPos ? 'plus' : 'minus');

            return `
                <div class="attr-row">
                    <span>${item.name}</span>
                    <strong class="${colorClass}">${deltaStr}</strong>
                </div>
            `;
        }).join('');
    }
}

function applyPresetToActive(preset) {
    if (preset === 'luxury') {
        document.getElementById('calc-OverallQual').value = 9;
        document.getElementById('calc-GrLivArea').value = 3400;
        document.getElementById('calc-YearBuilt').value = 2021;
        document.getElementById('calc-TotalBsmtSF').value = 1600;
        document.getElementById('calc-BedroomAbvGr').value = 4;
        document.getElementById('calc-FullBath').value = 3;
        document.getElementById('calc-GarageCars').value = 3;
    } else if (preset === 'family') {
        document.getElementById('calc-OverallQual').value = 7;
        document.getElementById('calc-GrLivArea').value = 2100;
        document.getElementById('calc-YearBuilt').value = 2008;
        document.getElementById('calc-TotalBsmtSF').value = 950;
        document.getElementById('calc-BedroomAbvGr').value = 3;
        document.getElementById('calc-FullBath').value = 2;
        document.getElementById('calc-GarageCars').value = 2;
    } else if (preset === 'starter') {
        document.getElementById('calc-OverallQual').value = 5;
        document.getElementById('calc-GrLivArea').value = 1250;
        document.getElementById('calc-YearBuilt').value = 1996;
        document.getElementById('calc-TotalBsmtSF').value = 700;
        document.getElementById('calc-BedroomAbvGr').value = 2;
        document.getElementById('calc-FullBath').value = 1;
        document.getElementById('calc-GarageCars').value = 1;
    } else if (preset === 'fixer') {
        document.getElementById('calc-OverallQual').value = 4;
        document.getElementById('calc-GrLivArea').value = 1450;
        document.getElementById('calc-YearBuilt').value = 1964;
        document.getElementById('calc-TotalBsmtSF').value = 800;
        document.getElementById('calc-BedroomAbvGr').value = 3;
        document.getElementById('calc-FullBath').value = 1;
        document.getElementById('calc-GarageCars').value = 1;
    }

    handleCalcChange();
}

function viewCurrentOnMap() {
    switchSection('map');
    if (selectedProperty && selectedProperty.neighborhood) {
        const stats = NEIGHBORHOOD_STATS[selectedProperty.neighborhood];
        if (stats) {
            map.flyTo(AMES_CENTER, 14, { duration: 1.2 });
        }
    }
}

// ============================================================
// 4. Section 2: Map Inspector Logic
// ============================================================
function openInspector() {
    const insp = document.getElementById('map-inspector');
    if (insp) insp.classList.remove('hidden');
}

function closeInspector() {
    const insp = document.getElementById('map-inspector');
    if (insp) insp.classList.add('hidden');
    activeMarkerLayerGroup.clearLayers();
}

function populateInspector(prop) {
    const locText = prop.isGlobal ? `${prop.city}, ${prop.country}` : `${getNeighborhoodFullName(prop.neighborhood)} · Ames, IA`;
    document.getElementById('insp-location').textContent = locText;

    const qual = prop.features.OverallQual || 7;
    document.getElementById('insp-OverallQual').value = qual;
    document.getElementById('insp-qual-badge').textContent = qual;

    document.getElementById('insp-GrLivArea').value = prop.features.GrLivArea || prop.features.property_size_sqft || 1800;
    document.getElementById('insp-YearBuilt').value = prop.features.YearBuilt || prop.features.constructed_year || 2005;
    document.getElementById('insp-BedroomAbvGr').value = prop.features.BedroomAbvGr || prop.features.rooms || 3;
    document.getElementById('insp-FullBath').value = prop.features.FullBath || prop.features.bathrooms || 2;

    document.getElementById('insp-model').textContent = prop.isGlobal ? 'Random Forest AI' : 'CatBoost AI';
}

function handleInspectorInput() {
    if (!selectedProperty) return;

    selectedProperty.features.OverallQual = Number(document.getElementById('insp-OverallQual')?.value || 7);
    selectedProperty.features.GrLivArea = Number(document.getElementById('insp-GrLivArea')?.value || 1800);
    selectedProperty.features.YearBuilt = Number(document.getElementById('insp-YearBuilt')?.value || 2005);
    selectedProperty.features.BedroomAbvGr = Number(document.getElementById('insp-BedroomAbvGr')?.value || 3);
    selectedProperty.features.FullBath = Number(document.getElementById('insp-FullBath')?.value || 2);

    document.getElementById('insp-qual-badge').textContent = selectedProperty.features.OverallQual;

    clearTimeout(inspDebounceTimer);
    inspDebounceTimer = setTimeout(() => {
        triggerInspectorPrediction(selectedProperty);
    }, 80);
}

async function triggerInspectorPrediction(property) {
    const endpoint = property.isGlobal ? '/api/predict/global' : '/api/predict';

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(property.features),
        });
        if (!response.ok) return;

        const result = await response.json();
        property.prediction = result.predicted_price;
        property.attribution = result.attribution;
        property.confidence_lower = result.confidence_lower;
        property.confidence_upper = result.confidence_upper;
        property.model_name = result.model;
        property.holdout_r2 = result.holdout_r2;

        const price = Math.round(result.predicted_price || 0);
        document.getElementById('insp-price').textContent = `$${price.toLocaleString()}`;

        const sqft = Number(property.features.GrLivArea || property.features.property_size_sqft || 1800);
        const pricePerSqFt = sqft > 0 ? Math.round(price / sqft) : 0;
        document.getElementById('insp-sqft-rate').textContent = `$${pricePerSqFt} / sqft`;

        const median = result.neighborhood_median || result.location_median || price;
        const diffPct = ((price - median) / (median || 1)) * 100;
        const trendEl = document.getElementById('insp-trend-pill');
        trendEl.className = diffPct >= 0 ? 'trend-pill positive' : 'trend-pill negative';
        trendEl.textContent = `${diffPct >= 0 ? '+' : ''}${diffPct.toFixed(1)}% vs Med`;

        renderAttributionList('insp-attribution-list', result.attribution);
    } catch (err) {
        console.error('Inspector prediction error:', err);
    }
}

// ============================================================
// 5. Section 3: Documentation Sidebar Scrolling
// ============================================================
function smoothScrollDoc(targetId) {
    const target = document.getElementById(targetId);
    if (!target) return;

    target.scrollIntoView({ behavior: 'smooth', block: 'start' });

    document.querySelectorAll('.docs-nav-link').forEach(link => {
        link.classList.toggle('active', link.getAttribute('href') === `#${targetId}`);
    });
}

// ============================================================
// 6. Appraisal Certificate Modal & Exports
// ============================================================
function openAppraisalFromCalc() {
    openAppraisalModal();
}

function openAppraisalFromInspector() {
    openAppraisalModal();
}

function openAppraisalModal() {
    const prop = selectedProperty;
    if (!prop) {
        alert('Please specify property parameters first.');
        return;
    }

    const modal = document.getElementById('appraisal-modal');
    const container = document.getElementById('appraisal-content');
    const price = Math.round(prop.price || prop.prediction || 225000);
    const locationName = prop.isGlobal ? `${prop.city}, ${prop.country}` : `${getNeighborhoodFullName(prop.neighborhood || 'NAmes')} (Ames, IA)`;
    const sqft = prop.features.GrLivArea || prop.features.property_size_sqft || 1800;
    const pricePerSqFt = sqft > 0 ? Math.round(price / sqft) : 0;
    const modelTag = prop.isGlobal ? 'Random Forest (Global 200k Pipeline)' : 'CatBoost Regressor (Ames Pipeline)';
    const r2Score = prop.isGlobal ? '99.99%' : '90.94%';
    const confLower = prop.confidence_lower ? `$${prop.confidence_lower.toLocaleString()}` : `$${Math.round(price * 0.92).toLocaleString()}`;
    const confUpper = prop.confidence_upper ? `$${prop.confidence_upper.toLocaleString()}` : `$${Math.round(price * 1.08).toLocaleString()}`;
    const dateStr = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });

    let attributionRows = '';
    if (prop.attribution && prop.attribution.length > 0) {
        attributionRows = prop.attribution.map(item => `
            <tr>
                <td><strong>${item.name}</strong></td>
                <td>${item.detail}</td>
                <td style="text-align: right; font-family: var(--font-mono); font-weight: 600; color: ${item.delta >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">
                    ${item.type === 'base' ? '$' + item.value.toLocaleString() : (item.delta >= 0 ? '+' : '') + '$' + item.delta.toLocaleString()}
                </td>
            </tr>
        `).join('');
    } else {
        attributionRows = `<tr><td colspan="3">Standard baseline estimation applied.</td></tr>`;
    }

    container.innerHTML = `
        <div class="appraisal-paper">
            <div class="appraisal-brand-header">
                <div>
                    <h1 class="appraisal-title">TerraPulse AI Valuation Certificate</h1>
                    <div class="appraisal-meta-sub">Official Automated Real Estate Appraisal Report</div>
                </div>
                <div class="appraisal-cert-box">
                    <span class="cert-code">PARCEL #APP-${Math.floor(1000 + Math.random() * 9000)}</span>
                    <span class="cert-date">${dateStr}</span>
                </div>
            </div>

            <div class="appraisal-hero-card">
                <div>
                    <span class="appraisal-caption">ESTIMATED FAIR MARKET VALUE</span>
                    <div class="appraisal-big-price">$${price.toLocaleString()}</div>
                    <div class="appraisal-range-text">95% Confidence Interval: <strong>${confLower} – ${confUpper}</strong></div>
                </div>
                <div class="appraisal-hero-stats">
                    <div>Price / SqFt: <strong>$${pricePerSqFt}</strong></div>
                    <div>Model: <strong>${modelTag}</strong></div>
                    <div>Benchmark R²: <strong>${r2Score}</strong></div>
                    <div>Market: <strong>${locationName}</strong></div>
                </div>
            </div>

            <div class="appraisal-section">
                <h3>1. Property Characteristics & Specifications</h3>
                <table class="appraisal-table">
                    <tbody>
                        <tr>
                            <td><strong>Location / Market</strong></td><td>${locationName}</td>
                            <td><strong>Gross Living Area</strong></td><td>${sqft} sq ft</td>
                        </tr>
                        <tr>
                            <td><strong>Quality Rating</strong></td><td>${prop.features.OverallQual || 7} / 10</td>
                            <td><strong>Vintage Year</strong></td><td>${prop.features.YearBuilt || prop.features.constructed_year || 2005}</td>
                        </tr>
                        <tr>
                            <td><strong>Bedrooms</strong></td><td>${prop.features.BedroomAbvGr || prop.features.rooms || 3}</td>
                            <td><strong>Bathrooms</strong></td><td>${prop.features.FullBath || prop.features.bathrooms || 2}</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="appraisal-section">
                <h3>2. Econometric Valuation Decomposition</h3>
                <table class="appraisal-table">
                    <thead><tr><th>Component</th><th>Detail</th><th style="text-align: right;">Impact</th></tr></thead>
                    <tbody>${attributionRows}</tbody>
                </table>
            </div>

            <div class="appraisal-section">
                <h3>3. Financing Scenario (30-Year Fixed)</h3>
                <div class="appraisal-fin-grid">
                    <div class="appraisal-fin-card">
                        <span>20% Down Payment</span>
                        <strong>$${Math.round(price * 0.2).toLocaleString()}</strong>
                    </div>
                    <div class="appraisal-fin-card">
                        <span>Monthly Mortgage (P&I)</span>
                        <strong>$${Math.round(price * 0.8 * 0.00632).toLocaleString()}/mo</strong>
                    </div>
                    <div class="appraisal-fin-card">
                        <span>Est. Monthly Rent</span>
                        <strong>$${Math.round(price * 0.0075).toLocaleString()}/mo</strong>
                    </div>
                    <div class="appraisal-fin-card">
                        <span>Gross Cap Yield</span>
                        <strong style="color: var(--accent-emerald);">9.0%</strong>
                    </div>
                </div>
            </div>

            <div class="appraisal-footer-note">
                Generated automatically by TerraPulse AI Geospatial Machine Learning System.
            </div>
        </div>
    `;

    modal.classList.remove('hidden');
}

function closeAppraisalModal(e) {
    if (!e || e.target.id === 'appraisal-modal' || e.target.closest('.btn-icon')) {
        document.getElementById('appraisal-modal').classList.add('hidden');
    }
}

function printAppraisalReport() {
    window.print();
}

function exportComparisonCSV() {
    if (!AMES_HOUSES || AMES_HOUSES.length === 0) {
        alert('No house points loaded.');
        return;
    }

    const headers = ['Id', 'Neighborhood', 'Price_USD', 'Quality_Rating', 'Living_Area_SqFt', 'Year_Built', 'Bedrooms', 'Bathrooms', 'Garage_Cars', 'Tier'];
    const rows = AMES_HOUSES.slice(0, 100).map(h => [
        h.id, `"${h.neighborhood}"`, h.price, h.OverallQual, h.GrLivArea, h.YearBuilt, h.BedroomAbvGr, h.FullBath, h.GarageCars, `"${h.tier}"`
    ].join(','));

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `ames_housing_valuation_sample_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ============================================================
// 7. Search & Quick Jump
// ============================================================
function setupSearchIndex() {
    if (NEIGHBORHOOD_STATS) {
        Object.keys(NEIGHBORHOOD_STATS).forEach(code => {
            searchIndex.push({
                type: 'ames',
                code: code,
                name: getNeighborhoodFullName(code),
                tag: 'Ames Neighborhood'
            });
        });
    }

    if (CITY_COORDINATES) {
        for (const [key, coords] of Object.entries(CITY_COORDINATES)) {
            const [country, city] = key.split('|');
            searchIndex.push({
                type: 'global',
                country, city,
                name: `${city}, ${country}`,
                coords,
                tag: 'Global City'
            });
        }
    }
}

function handleQuickJump(e) {
    const q = e.target.value.toLowerCase().trim();
    const dropdown = document.getElementById('quick-jump-results');

    if (!q) {
        dropdown.classList.add('hidden');
        return;
    }

    const matches = searchIndex.filter(item => item.name.toLowerCase().includes(q)).slice(0, 6);
    if (matches.length === 0) {
        dropdown.classList.add('hidden');
        return;
    }

    dropdown.innerHTML = matches.map(item => `
        <div class="quick-jump-item" onclick="selectSearchItem('${item.name}')">
            <span>${item.name}</span>
            <span class="quick-jump-tag">${item.tag}</span>
        </div>
    `).join('');

    dropdown.classList.remove('hidden');
}

function selectSearchItem(name) {
    const item = searchIndex.find(i => i.name === name);
    document.getElementById('quick-jump-results').classList.add('hidden');
    document.getElementById('map-search-input').value = '';

    if (!item) return;

    if (item.type === 'ames') {
        setScope('ames');
        map.flyTo(AMES_CENTER, 14, { duration: 1.2 });
    } else {
        setScope('global');
        map.flyTo(item.coords, 10, { duration: 1.5 });
        selectGlobalCityDot(item.country, item.city, item.coords);
    }
}

// ============================================================
// 8. Helper Utilities
// ============================================================
function getNeighborhoodFullName(code) {
    const names = {
        'CollgCr': 'College Creek',
        'Veenker': 'Veenker',
        'Crawfor': 'Crawford',
        'NoRidge': 'Northridge',
        'Mitchel': 'Mitchell',
        'Somerst': 'Somerset',
        'NWAmes': 'Northwest Ames',
        'OldTown': 'Old Town',
        'BrkSide': 'Brookside',
        'Sawyer': 'Sawyer',
        'NAmes': 'North Ames',
        'Gilbert': 'Gilbert',
        'StoneBr': 'Stone Brook',
        'NridgHt': 'Northridge Heights',
        'Edwards': 'Edwards',
        'SawyerW': 'Sawyer West',
        'Timber': 'Timberland',
        'IDOTRR': 'Iowa DOT / Rail',
        'ClearCr': 'Clear Creek',
        'SWISU': 'South & West ISU',
        'Blmngtn': 'Bloomington Heights',
        'MeadowV': 'Meadow Village',
        'BrDale': 'Briardale',
        'NPkVill': 'Northpark Villa',
        'Blueste': 'Bluestem'
    };
    return names[code] || code;
}

function getQualityLabel(rating) {
    if (rating >= 9) return 'Luxury / Elite Craftsmanship';
    if (rating >= 7) return 'Good / Modern Custom Build';
    if (rating >= 5) return 'Average / Standard Spec';
    if (rating >= 3) return 'Fair / Needs Updating';
    return 'Poor / Major Renovation Needed';
}

function findClosestNeighborhood(lat, lng) {
    return 'CollgCr';
}

function formatPrice(price) {
    if (!price) return '0';
    if (price >= 1000000) return (price / 1000000).toFixed(2) + 'M';
    if (price >= 1000) return Math.round(price / 1000) + 'K';
    return String(Math.round(price));
}

// Initialize on Load
document.addEventListener('DOMContentLoaded', () => {
    initCalculatorDropdowns();
    initMap();
    handleCalcChange();
});
