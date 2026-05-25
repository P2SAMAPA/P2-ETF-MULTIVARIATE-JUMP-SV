import streamlit as st
import pandas as pd
import numpy as np
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Multivariate Jump-SV Model", layout="wide")

st.markdown("""
<style>
.hero-card {
    background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%);
    padding: 1.5rem;
    border-radius: 1rem;
    margin: 0.5rem;
    text-align: center;
    color: white;
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}
.hero-card h3 {
    font-size: 2rem;
    margin: 0;
    font-weight: bold;
}
.hero-card p {
    font-size: 1.2rem;
    margin: 0.5rem 0 0;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align: center;">📈 Jump-Diffusion with Stochastic Volatility (BNS 2004)</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Jump clustering + stochastic volatility | Expected jump return as alpha signal</p>', unsafe_allow_html=True)

st.sidebar.markdown("## 🧮 Jump-SV Model")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True, type="primary"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows evaluated:** {', '.join(map(str, config.WINDOWS))} days")
st.sidebar.markdown(f"**Particles:** {config.N_PARTICLES} | **Max iter:** {config.MAX_OPT_ITER}")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'jump_sv_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']

def display_universe(universe_name, uni_data, window_data, window_label):
    top3 = window_data["top_etfs"]
    norm_scores = window_data["all_scores_norm"]
    raw_scores = window_data["all_scores_raw"]
    st.markdown(f'<h2 style="font-size: 1.8rem; margin-top: 1rem;">{universe_name.replace("_", " ").title()} <span style="font-size: 0.9rem; background: #e0e0e0; padding: 0.2rem 0.8rem; border-radius: 20px;">{window_label}</span></h2>', unsafe_allow_html=True)

    cols = st.columns(3)
    for idx, etf in enumerate(top3):
        with cols[idx]:
            st.markdown(f"""
            <div class="hero-card">
                <h3>{etf['ticker']}</h3>
                <p>Jump‑SV score: {etf['jumpsv_score_norm']:.3f}</p>
                <p style="font-size:0.9rem;">raw expected jump return: {etf['raw_score']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
    with st.expander(f"Full ranking for {universe_name}"):
        df_full = pd.DataFrame(list(norm_scores.items()), columns=["Ticker", "Normalized Score"])
        df_full["Raw Score"] = df_full["Ticker"].apply(lambda t: raw_scores[t])
        df_full = df_full.sort_values("Normalized Score", ascending=False)
        st.dataframe(df_full, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["📊 Best Window (Auto)", "🔍 Choose Window (Manual)", "📈 Walk-Forward Backtest"])

with tab1:
    st.header("🚀 Top ETFs by Expected Jump Return (Auto Best Window)")
    with st.expander("📖 Interpretation", expanded=False):
        st.markdown("""
        - **Jump-diffusion with stochastic volatility** (Barndorff-Nielsen & Shephard 2004)
        - The model estimates: diffusive volatility (CIR), jump intensity λ, jump size mean μⱼ.
        - **Signal = λ × μⱼ** = expected return contribution from jumps.
        - Positive expected jump return suggests future upward jump risk / reward.
        - The best window is selected by the highest average backtest return (walk-forward) across all ETFs.
        """)
    for universe_name, uni_data in data["universes"].items():
        if not uni_data or not uni_data.get("all_windows"):
            st.warning(f"No window data for {universe_name}")
            continue
        best_data = uni_data.get("best_window_data")
        if best_data is None and uni_data["all_windows"]:
            best_data = uni_data["all_windows"][-1]
            win_label = f"window {best_data['window']}d (fallback)"
        elif best_data:
            win_label = f"best window {best_data['window']}d"
        else:
            st.warning(f"No data for {universe_name}")
            continue
        display_universe(universe_name, uni_data, best_data, win_label)

with tab2:
    st.header("🔍 Manual Window Selection")
    st.markdown("Choose a rolling window to inspect the expected jump return per ETF.")
    for universe_name, uni_data in data["universes"].items():
        if not uni_data or not uni_data.get("all_windows"):
            st.warning(f"No window data for {universe_name}")
            continue
        available_windows = [wd["window"] for wd in uni_data["all_windows"]]
        sel_win = st.selectbox(f"Window for {universe_name.replace('_', ' ').title()}", available_windows, key=f"manual_{universe_name}")
        win_data = next((wd for wd in uni_data["all_windows"] if wd["window"] == sel_win), None)
        if win_data:
            display_universe(universe_name, uni_data, win_data, f"window {sel_win}d")
        else:
            st.warning("No data for selected window.")

with tab3:
    st.header("📈 Walk‑Forward Backtest (Per‑ETF Average Next‑Day Return)")
    st.markdown("""
    For each window length, the model is estimated recursively on expanding/rolling windows:
    - On each day, fit the Jump‑SV model to the trailing `window` days for each ETF.
    - Select the top 3 ETFs by expected jump return (λ × μⱼ).
    - Record the **next day's return** for each selected ETF.
    - The table below shows, for each universe and window, the top 3 ETFs by their **average next‑day return** when selected.
    This is a true out‑of‑sample performance measure.
    """)
    for universe_name, uni_data in data["universes"].items():
        if not uni_data or not uni_data.get("all_windows"):
            continue
        st.subheader(universe_name.replace("_", " ").title())
        rows = []
        for wd in uni_data["all_windows"]:
            w = wd["window"]
            backtest_dict = wd.get("backtest_per_etf_avg_return", {})
            if not backtest_dict:
                continue
            sorted_by_backtest = sorted(backtest_dict.items(), key=lambda x: x[1], reverse=True)[:config.TOP_N]
            for ticker, avg_ret in sorted_by_backtest:
                rows.append({
                    "Window (days)": w,
                    "Ticker": ticker,
                    "Avg next‑day return (%)": f"{avg_ret*100:.4f}%"
                })
        if rows:
            df_backtest = pd.DataFrame(rows)
            st.dataframe(df_backtest, use_container_width=True)
        else:
            st.info("No backtest data available for this universe (maybe not enough history).")

st.sidebar.markdown("---")
st.sidebar.caption("Multivariate Jump-SV | Barndorff-Nielsen & Shephard (2004)")
