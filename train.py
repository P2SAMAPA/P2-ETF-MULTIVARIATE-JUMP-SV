import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
import config
import data_manager as dm
from jump_sv_model import JumpSVDiffusion

def normalize_scores(score_dict):
    scores = np.array(list(score_dict.values()))
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-12:
        return {k: 0.0 for k in score_dict}
    norm = (scores - min_s) / (max_s - min_s)
    return {ticker: float(norm[i]) for i, ticker in enumerate(score_dict.keys())}

def rolling_walkforward_backtest(returns_df, window_days, top_n=3):
    """
    Walk-forward: for each day t, estimate model on trailing window_days,
    compute score for each ETF, pick top_n, record next day return.
    Returns per-ETF average next-day return when selected.
    """
    n = len(returns_df)
    sum_returns = {}
    count = {}
    model = JumpSVDiffusion(n_particles=config.N_PARTICLES)
    for t in range(window_days, n - 1):
        window = returns_df.iloc[t - window_days : t]
        next_day = returns_df.iloc[t]
        scores = {}
        for ticker in window.columns:
            try:
                res = model.fit(window[ticker].values, max_iter=config.MAX_OPT_ITER)
                score = res['score']
            except:
                score = 0.0
            scores[ticker] = score
        # Normalise to rank
        norm = normalize_scores(scores)
        sorted_etfs = sorted(norm.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [ticker for ticker, _ in sorted_etfs[:top_n]]
        for ticker in top_etfs:
            ret = next_day[ticker]
            sum_returns[ticker] = sum_returns.get(ticker, 0.0) + ret
            count[ticker] = count.get(ticker, 0) + 1
    avg_returns = {ticker: sum_returns[ticker]/count[ticker] for ticker in sum_returns}
    return avg_returns

def run_for_window(returns, window_days):
    if len(returns) < window_days:
        return None
    ret_window = returns.iloc[-window_days:]
    model = JumpSVDiffusion(n_particles=config.N_PARTICLES)
    scores_raw = {}
    param_details = {}
    for ticker in ret_window.columns:
        try:
            res = model.fit(ret_window[ticker].values, max_iter=config.MAX_OPT_ITER)
            scores_raw[ticker] = res['score']
            param_details[ticker] = {k: v for k, v in res.items() if k != 'score'}
        except Exception as e:
            print(f"    Error fitting {ticker}: {e}")
            scores_raw[ticker] = 0.0
            param_details[ticker] = {}
    norm_scores = normalize_scores(scores_raw)
    sorted_norm = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
    top_etfs = [{"ticker": t, "jumpsv_score_norm": s, "raw_score": scores_raw[t]} for t, s in sorted_norm[:config.TOP_N]]
    return {
        "window": window_days,
        "top_etfs": top_etfs,
        "all_scores_raw": scores_raw,
        "all_scores_norm": norm_scores,
        "param_details": param_details
    }

def main():
    print("Loading master data...")
    dm.load_master_data()
    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "n_particles": config.N_PARTICLES,
        "max_iter": config.MAX_OPT_ITER,
        "universes": {}
    }
    for uni_name in config.UNIVERSES.keys():
        print(f"Processing {uni_name}...")
        returns = dm.get_universe_returns(uni_name)
        if returns.empty:
            print("  No data -> skipping")
            continue
        all_window_results = []
        # Compute per-window scores and store
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            out = run_for_window(returns, w)
            if out:
                all_window_results.append(out)
            else:
                print(f"    Failed for window {w}")
        # Now compute backtest (walk-forward) per-ETF averages for each window (optional: we can compute once per window)
        # We'll compute backtest for each window to get per-ETF average returns (for display in backtest tab)
        for wdata in all_window_results:
            w = wdata['window']
            print(f"  Backtest for window {w} (walk-forward)...")
            backtest_etf_avg = rolling_walkforward_backtest(returns, w, top_n=config.TOP_N)
            wdata['backtest_per_etf_avg_return'] = backtest_etf_avg
        # Find best window by highest average backtest return (across all ETFs' averages)
        best_avg = -np.inf
        best_window = None
        best_data = None
        for wdata in all_window_results:
            bt_vals = list(wdata['backtest_per_etf_avg_return'].values())
            if bt_vals:
                avg_bt = np.mean(bt_vals)
                if avg_bt > best_avg:
                    best_avg = avg_bt
                    best_window = wdata['window']
                    best_data = wdata
        results["universes"][uni_name] = {
            "best_window_by_backtest": best_window,
            "best_window_data": best_data,
            "all_windows": all_window_results
        }
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"output/jump_sv_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")
    api = HfApi(token=config.HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=out_file,
            path_in_repo=os.path.basename(out_file),
            repo_id=config.OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.OUTPUT_REPO}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
