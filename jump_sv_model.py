import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, gamma

class JumpSVDiffusion:
    """
    Bivariate jump-diffusion with stochastic volatility (CIR for variance).
    Model:
        dV_t = kappa (theta - V_t) dt + sigma_v sqrt(V_t) dB_t
        dX_t = mu dt + sqrt(V_t) dW_t + dJ_t
        d<B,W>_t = rho dt
        J_t: compound Poisson with intensity lambda, jump size N(mu_j, sigma_j^2)
    """
    def __init__(self, n_particles=200):
        self.n_particles = n_particles

    def simulate(self, params, n_steps, dt=1.0, return_vol=False):
        """Simulate one path of returns and latent volatility (for testing)."""
        mu, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j = params
        V = np.zeros(n_steps)
        X = np.zeros(n_steps)
        V[0] = theta  # initial vol
        X[0] = 0.0
        for t in range(1, n_steps):
            Z1 = np.random.normal()
            Z2 = np.random.normal()
            dB = Z1 * np.sqrt(dt)
            dW = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)
            # CIR Euler
            V[t] = V[t-1] + kappa * (theta - V[t-1]) * dt + sigma_v * np.sqrt(max(V[t-1], 0)) * dB
            V[t] = max(V[t], 1e-8)
            # Jump
            if np.random.rand() < lam * dt:
                J = np.random.normal(mu_j, sigma_j)
            else:
                J = 0.0
            X[t] = X[t-1] + mu * dt + np.sqrt(max(V[t-1], 0)) * dW + J
        ret = np.diff(X)
        if return_vol:
            return ret, V[1:]
        return ret

    def particle_filter_loglik(self, returns, params, dt=1.0):
        """
        Compute log-likelihood of observed returns given parameters using a bootstrap particle filter.
        Returns: log-likelihood (scalar)
        """
        mu, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j = params
        T = len(returns)
        # Initial particles for V0 ~ Gamma(a,b) chosen to match stationary distribution
        a = theta**2 / sigma_v**2  # shape
        b = theta / sigma_v**2      # rate
        V = np.random.gamma(a, 1/b, size=self.n_particles)
        V = np.maximum(V, 1e-6)
        loglik = 0.0
        weights = np.ones(self.n_particles) / self.n_particles

        for t in range(T):
            # Proposal for V_t given V_{t-1} (CIR transition with no innovation correlation for prediction)
            # We'll use Euler step with correlated innovations? We'll use the transition density of V.
            # For simplicity, we sample V_next from exact conditional distribution? Not exact. Use Euler.
            # Generate innovations for B and W consistent with correlation
            Z1 = np.random.normal(size=self.n_particles)
            Z2 = np.random.normal(size=self.n_particles)
            dB = Z1 * np.sqrt(dt)
            dW = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)

            V_pred = V + kappa * (theta - V) * dt + sigma_v * np.sqrt(np.maximum(V, 0)) * dB
            V_pred = np.maximum(V_pred, 1e-6)

            # Observation likelihood (mixture of normals)
            # r_t = mu*dt + sqrt(V) * dW + J, where J ~ (1-lam*dt)*0 + lam*dt*N(mu_j, sigma_j^2)
            # So conditional on V, the density is (1-lam*dt)*N(r_t | mu*dt, V*dt) + lam*dt*N(r_t | mu*dt+mu_j, V*dt+sigma_j^2)
            # We use dt=1 for daily
            r = returns[t]
            mean1 = mu
            var1 = V_pred
            mean2 = mu + mu_j
            var2 = V_pred + sigma_j**2
            like1 = norm.pdf(r, mean1, np.sqrt(var1))
            like2 = norm.pdf(r, mean2, np.sqrt(var2))
            lam_dt = lam  # since dt=1
            weight = (1 - lam_dt) * like1 + lam_dt * like2
            weight = np.maximum(weight, 1e-300)
            # Update weights
            weights *= weight
            # Normalise
            total = np.sum(weights)
            loglik += np.log(total)
            weights /= total
            # Resample if effective sample size < threshold
            ess = 1.0 / np.sum(weights**2)
            if ess < self.n_particles / 2:
                indices = np.random.choice(self.n_particles, size=self.n_particles, p=weights)
                V = V_pred[indices]
                weights = np.ones(self.n_particles) / self.n_particles
            else:
                V = V_pred

        return loglik

    def fit(self, returns, initial_params=None, max_iter=30):
        """
        Maximum likelihood estimation using L-BFGS-B.
        Returns: dict of estimated params, score = lam * mu_j, and log-likelihood.
        """
        if initial_params is None:
            # Reasonable initial guesses
            mu0 = np.mean(returns)
            sig2 = np.var(returns)
            # Rough initial for SV
            kappa0 = 0.1
            theta0 = sig2
            sigma_v0 = 0.5 * sig2
            rho0 = 0.0
            lam0 = 0.05
            mu_j0 = 0.0
            sigma_j0 = np.std(returns) * 0.5
            initial_params = [mu0, kappa0, theta0, sigma_v0, rho0, lam0, mu_j0, sigma_j0]

        bounds = [(None, None), (1e-5, 10), (1e-5, None), (1e-5, None), (-0.99, 0.99), (1e-5, 0.5), (None, None), (1e-5, None)]
        # Wrap objective
        def obj(params):
            try:
                ll = self.particle_filter_loglik(returns, params)
                return -ll  # minimize negative log-likelihood
            except:
                return 1e10

        res = minimize(obj, initial_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})
        params_opt = res.x
        mu, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j = params_opt
        score = lam * mu_j   # expected jump return
        return {
            'mu': mu,
            'kappa': kappa,
            'theta': theta,
            'sigma_v': sigma_v,
            'rho': rho,
            'lambda': lam,
            'mu_j': mu_j,
            'sigma_j': sigma_j,
            'score': score,
            'loglik': -res.fun
        }
