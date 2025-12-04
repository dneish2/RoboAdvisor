"""OptionSim Pro Streamlit component."""

import json
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit import components


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_greek_lookup(option_chain_df: pd.DataFrame, option_type: str) -> Dict[Tuple[str, float], Dict]:
    lookup: Dict[Tuple[str, float], Dict] = {}
    if option_chain_df is None or option_chain_df.empty:
        return lookup

    for _, row in option_chain_df.iterrows():
        strike = _safe_float(row.get("strike"))
        if strike is None:
            continue
        lookup[(option_type, strike)] = {
            "delta": row.get("delta"),
            "theta": row.get("theta"),
            "vega": row.get("vega"),
            "impliedVolatility": row.get("impliedVolatility"),
            "lastPrice": row.get("lastPrice"),
        }
    return lookup


def _normalize_option_rows(options_df: pd.DataFrame, greek_lookup: Dict[Tuple[str, float], Dict], expiration: str):
    normalized = []
    for _, row in options_df.iterrows():
        strike = _safe_float(row.get("Strike"))
        option_type = str(row.get("Type", "")).upper()
        greek_row = greek_lookup.get((option_type, strike), {})

        normalized.append(
            {
                "optionType": option_type,
                "strike": strike,
                "lastPrice": _safe_float(row.get("Last Price"))
                if pd.notna(row.get("Last Price"))
                else _safe_float(greek_row.get("lastPrice")),
                "delta": _safe_float(greek_row.get("delta")),
                "theta": _safe_float(greek_row.get("theta")),
                "vega": _safe_float(greek_row.get("vega")),
                "impliedVolatility": _safe_float(greek_row.get("impliedVolatility")),
                "expiration": expiration,
            }
        )

    return normalized


def render_option_sim(data_fetcher):
    st.header("OptionSim Pro")

    if "options_data" not in st.session_state or st.session_state.get("options_data") is None:
        st.info("Fetch options data from Instrument Scanner to enable OptionSim Pro.")
        return

    cached_options = st.session_state.get("options_data")
    options_df = cached_options.copy() if isinstance(cached_options, pd.DataFrame) else pd.DataFrame(cached_options)

    if options_df.empty:
        st.info("Fetch options data from Instrument Scanner to enable OptionSim Pro.")
        return

    ticker = st.session_state.get("options_ticker") or ""
    if not ticker:
        st.info("Ticker context is missing. Please refetch options data.")
        return

    if pd.api.types.is_datetime64_any_dtype(options_df.get("Expiration")):
        options_df["Expiration"] = options_df["Expiration"].dt.strftime("%Y-%m-%d")
    else:
        options_df["Expiration"] = options_df["Expiration"].astype(str)

    available_expirations = sorted(options_df["Expiration"].unique().tolist())
    if not available_expirations:
        st.warning("No expiration dates available from cached options data.")
        return

    selected_expiration = st.selectbox("Select Expiration", available_expirations)

    strikes = data_fetcher.get_strikes_for_expiration(ticker, selected_expiration)
    if not strikes:
        st.warning("No strikes returned for the chosen expiration.")
        return

    default_strike = strikes[0]
    filtered_expiration_df = options_df[options_df["Expiration"] == selected_expiration]
    if not filtered_expiration_df.empty:
        try:
            closest_strike = min(strikes, key=lambda strike: abs(strike - filtered_expiration_df["Strike"].median()))
            default_strike = closest_strike
        except Exception:
            default_strike = strikes[0]

    selected_strike = st.selectbox("Select Strike", strikes, index=strikes.index(default_strike))

    filtered_options = filtered_expiration_df[filtered_expiration_df["Strike"] == selected_strike]
    if filtered_options.empty:
        st.warning("No options in session cache match the selected expiration and strike.")
        return

    ticker_obj = yf.Ticker(ticker)
    greek_lookup: Dict[Tuple[str, float], Dict] = {}
    try:
        option_chain = ticker_obj.option_chain(selected_expiration)
        greek_lookup.update(_build_greek_lookup(option_chain.calls, "CALL"))
        greek_lookup.update(_build_greek_lookup(option_chain.puts, "PUT"))
    except Exception:
        st.warning("Live greeks are unavailable for this expiration. Displaying cached pricing only.")

    normalized_options = _normalize_option_rows(filtered_options, greek_lookup, selected_expiration)
    if not normalized_options:
        st.warning("No options available to visualize.")
        return

    try:
        expiration_date = datetime.strptime(selected_expiration, "%Y-%m-%d")
        days_to_expiration = max((expiration_date.date() - datetime.utcnow().date()).days, 0)
    except Exception:
        days_to_expiration = None

    context = {
        "ticker": ticker,
        "expiration": selected_expiration,
        "daysToExpiration": days_to_expiration,
    }

    options_json = json.dumps(normalized_options)
    context_json = json.dumps(context)

    html_content = f"""
    <style>
        .options-sim-wrapper {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: radial-gradient(circle at 10% 20%, rgba(46, 54, 68, 0.6), rgba(15, 18, 26, 0.95)), #0f121a;
            color: #e8edf7;
            padding: 18px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .options-sim-header {{
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 12px;
        }}
        .options-sim-header h3 {{
            margin: 0;
            color: #7ad7ff;
        }}
        .context-pill {{
            background: rgba(122, 215, 255, 0.1);
            border: 1px solid rgba(122, 215, 255, 0.25);
            border-radius: 8px;
            padding: 6px 10px;
            color: #d4e7ff;
            font-size: 13px;
        }}
        table.options-grid {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
            background: rgba(255,255,255,0.01);
        }}
        table.options-grid th, table.options-grid td {{
            padding: 8px 6px;
            border-bottom: 1px solid rgba(255,255,255,0.07);
            text-align: center;
            font-size: 13px;
        }}
        table.options-grid th {{
            color: #a6b2c4;
            font-weight: 600;
        }}
        tr.primary-row {{
            background: linear-gradient(90deg, rgba(122, 215, 255, 0.12), rgba(122, 215, 255, 0));
        }}
        tr.selected-row {{
            outline: 1px solid rgba(122, 215, 255, 0.35);
        }}
        .delta-positive {{ color: #7CFFCB; }}
        .delta-negative {{ color: #ff9e9e; }}
        .muted {{ color: #9aa6b9; }}
        .summary-panel {{
            margin-top: 12px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }}
        .summary-card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 10px;
            padding: 10px 12px;
        }}
        #pl-chart {{
            margin-top: 16px;
        }}
    </style>
    <div class="options-sim-wrapper">
        <div class="options-sim-header">
            <h3>OptionSim Pro</h3>
            <div class="context-pill">Ticker: {context['ticker']} | Expiration: {context['expiration']} | Days to Expiry: {context.get('daysToExpiration', 'N/A')}</div>
        </div>
        <table class="options-grid">
            <thead>
                <tr>
                    <th>Primary</th>
                    <th>Select</th>
                    <th>Type</th>
                    <th>Strike</th>
                    <th>Last</th>
                    <th>Delta</th>
                    <th>Theta</th>
                    <th>Vega</th>
                    <th>IV</th>
                </tr>
            </thead>
            <tbody id="options-body"></tbody>
        </table>
        <div class="summary-panel">
            <div class="summary-card">
                <div class="muted">Primary Contract</div>
                <div id="primary-label">None selected</div>
            </div>
            <div class="summary-card">
                <div class="muted">Multi-Select</div>
                <div><span id="selected-count">0</span> contracts active</div>
            </div>
            <div class="summary-card">
                <div class="muted">Average IV</div>
                <div id="iv-avg">N/A</div>
            </div>
            <div class="summary-card">
                <div class="muted">Delta Exposure</div>
                <div id="delta-sum">N/A</div>
            </div>
        </div>
        <canvas id="pl-chart"></canvas>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const optionsData = {options_json};
        const contextData = {context_json};
        let primaryIndex = null;
        let selectedRows = new Set();
        let chartInstance = null;

        const formatNumber = (value, digits = 2) => {
            if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
            return Number(value).toFixed(digits);
        };

        function renderTable() {
            const body = document.getElementById('options-body');
            body.innerHTML = '';
            optionsData.forEach((opt, idx) => {
                const row = document.createElement('tr');
                if (primaryIndex === idx) row.classList.add('primary-row');
                if (selectedRows.has(idx)) row.classList.add('selected-row');

                const primaryCell = document.createElement('td');
                const primaryRadio = document.createElement('input');
                primaryRadio.type = 'radio';
                primaryRadio.name = 'primary-option';
                primaryRadio.checked = primaryIndex === idx;
                primaryRadio.onclick = () => {
                    primaryIndex = idx;
                    if (!selectedRows.has(idx)) selectedRows.add(idx);
                    updateSummary();
                    renderTable();
                };
                primaryCell.appendChild(primaryRadio);
                row.appendChild(primaryCell);

                const selectCell = document.createElement('td');
                const selectBox = document.createElement('input');
                selectBox.type = 'checkbox';
                selectBox.checked = selectedRows.has(idx);
                selectBox.onchange = (evt) => {
                    if (evt.target.checked) {
                        selectedRows.add(idx);
                    } else {
                        selectedRows.delete(idx);
                        if (primaryIndex === idx) primaryIndex = null;
                    }
                    updateSummary();
                    renderTable();
                };
                selectCell.appendChild(selectBox);
                row.appendChild(selectCell);

                const cells = [
                    opt.optionType || '—',
                    formatNumber(opt.strike),
                    formatNumber(opt.lastPrice),
                    formatNumber(opt.delta),
                    formatNumber(opt.theta),
                    formatNumber(opt.vega),
                    formatNumber(opt.impliedVolatility, 4)
                ];

                cells.forEach((value, cIdx) => {
                    const cell = document.createElement('td');
                    if (cIdx === 3) {
                        if (opt.delta !== null && opt.delta !== undefined) {
                            cell.classList.add(opt.delta >= 0 ? 'delta-positive' : 'delta-negative');
                        } else {
                            cell.classList.add('muted');
                        }
                    }
                    cell.textContent = value;
                    row.appendChild(cell);
                });

                body.appendChild(row);
            });
            updateSummary();
        }

        function updateSummary() {
            document.getElementById('selected-count').textContent = selectedRows.size;
            const primaryLabel = document.getElementById('primary-label');
            if (primaryIndex !== null) {
                const opt = optionsData[primaryIndex];
                primaryLabel.textContent = `${{opt.optionType}} ${{opt.strike}}`;
            } else {
                primaryLabel.textContent = 'None selected';
            }

            const selected = Array.from(selectedRows).map(idx => optionsData[idx]);
            const ivs = selected.map(o => o.impliedVolatility).filter(v => v !== null && v !== undefined);
            const avgIv = ivs.length ? ivs.reduce((a,b)=>a+b,0)/ivs.length : null;
            document.getElementById('iv-avg').textContent = avgIv !== null ? formatNumber(avgIv * 100, 2) + '%' : 'N/A';

            const deltaSum = selected.map(o => o.delta || 0).reduce((a,b)=>a+b,0);
            document.getElementById('delta-sum').textContent = selected.length ? formatNumber(deltaSum, 3) : 'N/A';

            renderChart();
        }

        function renderChart() {
            const ctx = document.getElementById('pl-chart').getContext('2d');
            if (chartInstance) chartInstance.destroy();

            if (primaryIndex === null) {
                chartInstance = null;
                ctx.font = '14px sans-serif';
                ctx.fillStyle = '#9aa6b9';
                ctx.fillText('Select a primary contract to view projected P/L.', 10, 20);
                return;
            }

            const primary = optionsData[primaryIndex];
            const pctMoves = [-10, -5, 0, 5, 10];
            const base = Number(primary.lastPrice) || 0;
            const delta = Number(primary.delta) || 0;
            const theta = Number(primary.theta) || 0;
            const days = contextData.daysToExpiration || 0;
            const plPoints = pctMoves.map(pct => base + delta * pct + (theta / 365) * days);

            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: pctMoves.map(p => `${{p}}% move`),
                    datasets: [{
                        label: 'Projected Premium',
                        data: plPoints,
                        borderColor: '#7ad7ff',
                        backgroundColor: 'rgba(122, 215, 255, 0.25)',
                        tension: 0.25,
                        fill: true,
                    }]
                },
                options: {
                    plugins: {
                        legend: { labels: { color: '#e8edf7' } }
                    },
                    scales: {
                        x: { ticks: { color: '#9aa6b9' }, grid: { color: 'rgba(255,255,255,0.06)' } },
                        y: { ticks: { color: '#9aa6b9' }, grid: { color: 'rgba(255,255,255,0.06)' } }
                    }
                }
            });
        }

        renderTable();
    </script>
    """

    components.v1.html(html_content, height=800, scrolling=True)
