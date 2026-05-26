import numpy as np
from scipy.optimize import minimize_scalar

class JumpSVDiffusion:
    """
    Fast approximate Jump-diffusion with stochastic volatility using threshold detection.
    """
    def __init__(self, n_particles=None):  # kept for compatibility
        pass

    def fit(self, returns, max_iter=None, **kwargs):
        """
        Estimate jump parameters using threshold method.
        Returns dict: score = expected jump return (positive jumps minus negative jumps)
        """
        ret = np.array(returns).flatten()
        if len(ret) < 5:
            return {'score': 0.0, 'lambda': 0.0, 'mu_j': 0.0}
        # Rolling volatility (EWMA or simple)
        vol = np.sqrt(np.convolve(ret**2, np.ones(20)/20, mode='same'))
        vol = np.maximum(vol, 1e-6)
        # Threshold = 2 * vol (adaptive)
        threshold = 2.0 * vol
        # Detect jumps
        pos_jump = ret > threshold
        neg_jump = ret < -threshold
        n_pos = np.sum(pos_jump)
        n_neg = np.sum(neg_jump)
        lam_pos = n_pos / len(ret)
        lam_neg = n_neg / len(ret)
        mu_j_pos = np.mean(ret[pos_jump]) if n_pos > 0 else 0.0
        mu_j_neg = np.mean(ret[neg_jump]) if n_neg > 0 else 0.0
        # Expected jump return = positive contribution - negative contribution
        score = lam_pos * mu_j_pos - lam_neg * mu_j_neg
        return {
            'score': score,
            'lambda_pos': lam_pos,
            'lambda_neg': lam_neg,
            'mu_j_pos': mu_j_pos,
            'mu_j_neg': mu_j_neg,
            'threshold': np.mean(threshold)
        }
