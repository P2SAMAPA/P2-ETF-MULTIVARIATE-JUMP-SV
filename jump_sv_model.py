import numpy as np
from scipy.optimize import minimize

class JumpSVDiffusion:
    """
    GARCH(1,1) with jump detection via standardised residuals.
    Estimated via MLE (GARCH) + thresholding on residuals.
    """
    def __init__(self, n_particles=None):
        pass

    @staticmethod
    def _garch_ll(params, ret):
        omega, alpha, beta = params
        omega = np.exp(omega)          # positivity
        alpha = 1 / (1 + np.exp(-alpha))  # 0..1
        beta = 1 / (1 + np.exp(-beta))    # 0..1
        if alpha + beta >= 1:
            return 1e10
        n = len(ret)
        var = np.zeros(n)
        var[0] = np.var(ret)
        for t in range(1, n):
            var[t] = omega + alpha * ret[t-1]**2 + beta * var[t-1]
        ll = -0.5 * np.sum(np.log(2*np.pi*var) + ret**2 / var)
        return -ll   # negative for minimisation

    def fit(self, returns, max_iter=30):
        ret = np.array(returns).flatten()
        if len(ret) < 10:
            return {'score': 0.0, 'lambda': 0.0, 'mu_j': 0.0}

        # Estimate GARCH(1,1) parameters
        init = [np.log(np.var(ret)*0.1), 0.1, 0.8]
        bounds = [(-10, 10), (-5, 5), (-5, 5)]
        res = minimize(self._garch_ll, init, args=(ret,),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter})
        omega = np.exp(res.x[0])
        alpha = 1 / (1 + np.exp(-res.x[1]))
        beta = 1 / (1 + np.exp(-res.x[2]))

        # Compute conditional variances
        n = len(ret)
        var = np.zeros(n)
        var[0] = np.var(ret)
        for t in range(1, n):
            var[t] = omega + alpha * ret[t-1]**2 + beta * var[t-1]

        # Standardised residuals
        resid = ret / np.sqrt(var)

        # Jump detection: residuals outside ±2.5 (standard choice)
        threshold = 2.5
        pos_jump = resid > threshold
        neg_jump = resid < -threshold
        lam_pos = np.mean(pos_jump)
        lam_neg = np.mean(neg_jump)
        mu_j_pos = np.mean(ret[pos_jump]) if np.any(pos_jump) else 0.0
        mu_j_neg = np.mean(ret[neg_jump]) if np.any(neg_jump) else 0.0

        # Expected jump return = positive contribution minus negative
        score = lam_pos * mu_j_pos - lam_neg * mu_j_neg

        return {
            'score': score,
            'lambda_pos': lam_pos,
            'lambda_neg': lam_neg,
            'mu_j_pos': mu_j_pos,
            'mu_j_neg': mu_j_neg,
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'threshold': threshold
        }
