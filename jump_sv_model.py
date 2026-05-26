import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from numpy.linalg import inv

class JumpSVDiffusion:
    """
    Jump-diffusion with stochastic volatility (CIR) + Bernoulli-Gaussian jumps.
    Estimated via:
      1. QMLE for CIR parameters using Kalman filter on log-squared returns.
      2. EM algorithm for jump parameters (Bernoulli-Gaussian mixture).
    """
    def __init__(self, n_particles=None):
        # n_particles kept for API compatibility; not used
        pass

    # ---------- Helper: CIR Kalman filter on log-volatility ----------
    @staticmethod
    def _cir_kalman(returns, kappa, theta, sigma_v, rho, mu, dt=1.0):
        """
        Kalman filter on log(V_t) using state-space model.
        Returns: filtered log-variance and log-likelihood.
        """
        n = len(returns)
        # State-space representation for log(V_t)
        # Transition: log(V_t) = (1 - kappa*dt)*log(V_{t-1}) + kappa*dt*log(theta) + innovation
        # Observation: log(r_t^2) = log(V_t) + log(z_t^2) where z_t ~ N(0,1) (but with leverage)
        # We use approximate linearisation
        log_ret2 = np.log(np.maximum(returns**2, 1e-12))
        # Initial state
        logV = np.log(theta)
        P = 1.0
        ll = 0.0
        logV_filt = np.zeros(n)
        for t in range(n):
            # Prediction
            logV_pred = (1 - kappa*dt) * logV + kappa*dt*np.log(theta)
            P_pred = (1 - kappa*dt)**2 * P + sigma_v**2 * dt
            # Update
            H = 1.0  # observation matrix
            R = np.pi**2 / 2  # variance of log(chi^2_1)
            K = P_pred * H / (H * P_pred * H + R)
            logV = logV_pred + K * (log_ret2[t] - logV_pred)
            P = (1 - K*H) * P_pred
            # Likelihood contribution
            innov = log_ret2[t] - logV_pred
            var_innov = P_pred + R
            ll += -0.5 * (np.log(2*np.pi*var_innov) + innov**2 / var_innov)
            logV_filt[t] = logV
        return logV_filt, ll

    # ---------- EM for Bernoulli-Gaussian jumps ----------
    @staticmethod
    def _em_jump_params(returns, logV):
        """
        Estimate jump intensity and jump size distribution using EM on mixture:
        P(return | V) = (1-λ)*N(0, V*dt) + λ*N(μ_j, V*dt + σ_j^2)
        """
        dt = 1.0
        V = np.exp(logV)
        n = len(returns)
        # Initial parameters
        lam = 0.05
        mu_j = 0.0
        sigma_j = np.std(returns) * 0.5
        # EM iterations
        for _ in range(20):
            # E-step: compute posterior probability of jump
            like_no_jump = norm.pdf(returns, 0, np.sqrt(V*dt))
            like_jump = norm.pdf(returns, mu_j, np.sqrt(V*dt + sigma_j**2))
            denom = (1-lam)*like_no_jump + lam*like_jump
            p_jump = lam * like_jump / denom
            # M-step
            lam_new = np.mean(p_jump)
            if lam_new > 0 and lam_new < 1:
                w = p_jump / np.sum(p_jump)
                mu_j_new = np.sum(w * returns)
                sigma_j2_new = np.sum(w * (returns - mu_j_new)**2)
                sigma_j_new = np.sqrt(sigma_j2_new)
            else:
                mu_j_new = mu_j
                sigma_j_new = sigma_j
            # Update
            lam = lam_new
            mu_j = mu_j_new
            sigma_j = sigma_j_new
        expected_jump_return = lam * mu_j
        return expected_jump_return, lam, mu_j, sigma_j

    # ---------- Main fit function ----------
    def fit(self, returns, max_iter=30):
        """
        Fit the full model via QMLE (CIR) + EM (jumps).
        Returns: dict with score = expected jump return.
        """
        ret = np.array(returns).flatten()
        if len(ret) < 10:
            return {'score': 0.0, 'lambda': 0.0, 'mu_j': 0.0}

        # Objective for CIR parameters (negative log-likelihood from Kalman)
        def neg_loglik_cir(params):
            kappa, theta, sigma_v, rho, mu = params
            kappa = np.exp(kappa)      # ensure positivity
            theta = np.exp(theta)
            sigma_v = np.exp(sigma_v)
            rho = np.tanh(rho)          # -1..1
            # mu free (no constraint)
            _, ll = self._cir_kalman(ret, kappa, theta, sigma_v, rho, mu)
            return -ll

        # Initial guesses (log-scale for positive params)
        initial = [np.log(0.1), np.log(np.var(ret)), np.log(0.2), 0.0, np.mean(ret)]
        bounds = [(-5, 2), (-5, 2), (-5, 2), (-3, 3), (-0.1, 0.1)]
        res = minimize(neg_loglik_cir, initial, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter})
        opt_params = res.x
        kappa = np.exp(opt_params[0])
        theta = np.exp(opt_params[1])
        sigma_v = np.exp(opt_params[2])
        rho = np.tanh(opt_params[3])
        mu = opt_params[4]

        # Get filtered log-volatility
        logV_filt, _ = self._cir_kalman(ret, kappa, theta, sigma_v, rho, mu)

        # Estimate jump parameters via EM
        score, lam, mu_j, sigma_j = self._em_jump_params(ret, logV_filt)

        return {
            'score': score,
            'lambda': lam,
            'mu_j': mu_j,
            'sigma_j': sigma_j,
            'kappa': kappa,
            'theta': theta,
            'sigma_v': sigma_v,
            'rho': rho,
            'mu': mu
        }
