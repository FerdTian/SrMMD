import jax
import jax.numpy as jnp


class BayesianLogisticRegressionTarget:
    def __init__(self, X_train, y_train, alpha_shape, alpha_rate):
        X_train = jnp.asarray(X_train, dtype=jnp.float64)
        y_train = jnp.asarray(y_train, dtype=jnp.float64)
        bias = jnp.ones((X_train.shape[0], 1), dtype=X_train.dtype)

        self.X_aug = jnp.concatenate([X_train, bias], axis=1)
        self.y_train = y_train
        self.alpha_shape = float(alpha_shape)
        self.alpha_rate = float(alpha_rate)
        self.weight_dim = int(self.X_aug.shape[1])
        self.latent_dim = self.weight_dim + 1

        self._score_single = jax.jit(jax.grad(self.log_posterior))

    @staticmethod
    def _safe_alpha(log_alpha):
        return jnp.exp(jnp.clip(log_alpha, -20.0, 20.0))

    def log_posterior(self, latent):
        weights = latent[:-1]
        log_alpha = latent[-1]
        alpha = self._safe_alpha(log_alpha)
        logits = self.X_aug @ weights

        log_likelihood = -jnp.sum(jnp.logaddexp(0.0, logits) - self.y_train * logits)
        log_prior_w_given_alpha = 0.5 * self.weight_dim * log_alpha - 0.5 * alpha * jnp.dot(weights, weights)
        log_prior_alpha = (self.alpha_shape - 1.0) * log_alpha - self.alpha_rate * alpha
        log_jacobian = log_alpha
        return log_likelihood + log_prior_w_given_alpha + log_prior_alpha + log_jacobian

    def score(self, latent):
        latent = jnp.asarray(latent, dtype=jnp.float64)
        if latent.ndim == 1:
            return self._score_single(latent)
        return jax.vmap(self._score_single)(latent)

    def sample_initial_particles(self, rng_key, particle_num, weight_scale=0.1, log_alpha_scale=0.5):
        weight_key, alpha_key = jax.random.split(rng_key)
        weights = weight_scale * jax.random.normal(weight_key, shape=(particle_num, self.weight_dim))
        alpha_center = jnp.log(jnp.maximum(self.alpha_shape / self.alpha_rate, 1e-6))
        log_alpha = alpha_center + log_alpha_scale * jax.random.normal(alpha_key, shape=(particle_num, 1))
        return jnp.concatenate([weights, log_alpha], axis=1)
