# Multivariate Jump-Diffusion with Stochastic Volatility (BNS 2004)

Implements the Barndorff-Nielsen & Shephard (2004) model: daily returns = diffusive component (CIR stochastic volatility) + compound Poisson jumps. The engine estimates parameters via particle filter maximum likelihood and produces a score = expected jump return (λ × μⱼ). This is a full‑fledged model combining jump clustering and stochastic volatility, absent from your current suite.

## Features
- Three ETF universes
- Seven rolling windows (63–4536 days)
- Latent volatility CIR process with leverage effect (correlation)
- Jump intensity and jump size parameters
- Particle filter (sequential Monte Carlo) for likelihood evaluation
- Walk‑forward backtest: select top 3 by expected jump return, measure next‑day return
- Two‑tab + backtest Streamlit dashboard
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-multivariate-jump-sv-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (may take a while due to particle filter, but runs overnight via GitHub Actions)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High expected jump return (λ × μⱼ) suggests the ETF is likely to experience a positive jump tomorrow.
- The walk‑forward backtest validates the signal's predictive power for next‑day returns.

## Requirements

See `requirements.txt`.
