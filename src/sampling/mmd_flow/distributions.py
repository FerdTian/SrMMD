import jax.numpy as jnp
import jax
import scipy
from functools import partial
from tqdm import tqdm
import jax.scipy as jsp  # 确保有这个

class Distribution:
    def __init__(self, kernel, means, covariances, integrand_name, weights=None):
        """
        A class that supports Gaussian and Mixture of Gaussians distributions.

        Parameters:
        - kernel: the kernel
        - means: (d,) array for a single Gaussian mean, or (k, d) for MoG.
        - covariances: (d, d) for a single Gaussian, or (k, d, d) for MoG.
        - weights: (k,) array for MoG. If None, assumes a single Gaussian.
        """
        self.kernel = kernel
        self.means = jnp.atleast_2d(means)  # Ensure shape (k, d)
        self.covariances = jnp.atleast_3d(covariances)  # Ensure shape (k, d, d)
        self.k, self.d = self.means.shape
        self.integrand_name = integrand_name
        if integrand_name == 'square':
            self.integrand = lambda x: (x**2).sum(1)
        elif integrand_name == 'neg_exp':
            self.integrand = lambda x: jnp.exp(-(x**2).sum(1) / (self.d ** 2 / 2))
        else:
            raise ValueError('Function not recognized!')
        
        if weights is None:
            self.weights = jnp.array([1.0])  # Single Gaussian case
        else:
            self.weights = jnp.asarray(weights)
            assert len(self.weights) == self.k, "Weights must match number of components."
            assert jnp.isclose(self.weights.sum(), 1), "Weights must sum to 1."

    def mean_embedding(self, Y):
        # Stein kernel case: E_{X~p}[k_p(X, y)] = 0
        if getattr(self.kernel, "is_stein", False):
            if Y.ndim == 1:
                return jnp.array(0.0)
            else:
                return jnp.zeros(Y.shape[:-1])
        # Vectorized computation using vmap
        kme_values = jax.vmap(self.kernel.mean_embedding, in_axes=(None, 0, 0))(Y, self.means, self.covariances)
        kme = jnp.tensordot(self.weights, kme_values, axes=1)
        return kme
    
    def mean_mean_embedding(self):
        if getattr(self.kernel, "is_stein", False):
            return jnp.array(0.0)
        if self.k == 1:
            double_kme = self.kernel.mean_mean_embedding(
                self.means[0], self.means[0], self.covariances[0], self.covariances[0]
            )
            return double_kme
        else:
            double_kme = 0
            for i in range(self.k):
                for j in range(self.k):
                    double_kme += self.weights[i] * self.weights[j] * self.kernel.mean_mean_embedding(self.means[i], self.means[j], self.covariances[i], self.covariances[j])
            return double_kme
    
    def sample(self, sample_size, rng_key):
        """
        Sample i.i.d from the mixture of Gaussians.

        Parameters:
        - sample_size: int, the number of samples to draw.
        - rng_key: JAX PRNGKey for reproducibility.

        Returns:
        - samples: (sample_size, d) array of samples.
        """
        rng_key, _ = jax.random.split(rng_key)
        component_indices = jax.random.choice(rng_key, self.k, shape=(sample_size,), p=self.weights)

        means = self.means[component_indices, :]
        covs = self.covariances[component_indices, :, :]

        def sample_gaussian(mean, cov, key):
            return jax.random.multivariate_normal(key, mean, cov)

        subkeys = jax.random.split(rng_key, sample_size)
        samples = jax.vmap(sample_gaussian)(means, covs, subkeys)
        return samples
    
    def qmc_sample(self, sample_size, rng_key):
        """
        Sample QMC from the mixture of Gaussians.

        Parameters:
        - sample_size: int, the number of samples to draw.
        - rng_key: JAX PRNGKey for reproducibility.

        Returns:
        - samples: (sample_size, d) array of samples.
        """
        component_indices = jax.random.choice(rng_key, self.k, shape=(sample_size,), p=self.weights)
        unique_components, sample_sizes = jnp.unique(component_indices, return_counts=True)

        mean = self.means[unique_components]
        cov = self.covariances[unique_components]

        def generate_qmc_samples(mean, cov, size):
            sobol = scipy.stats.qmc.Sobol(self.d)
            u = jnp.array(sobol.random(size))  # Generate Sobol sequence
            L = jnp.linalg.cholesky(cov)      # Compute Cholesky decomposition
            return mean + jax.scipy.stats.norm.ppf(u) @ L.T

        # Generate samples for each unique Gaussian component
        samples_dict = {
            int(unique_components[i]): generate_qmc_samples(mean[i], cov[i], sample_sizes[i])
            for i in range(len(unique_components))
        }
        qmc_samples = jnp.concatenate([samples_dict[int(idx)] for idx in samples_dict.keys()], axis=0)
        return qmc_samples
    
    def pdf(self, Y):
        """
        Compute the probability density function of the mixture of Gaussians.

        Parameters:
        - Y: (n, d) array of points to evaluate the PDF at.

        Returns:
        - pdf: (n,) array of PDF values.
        """
        pdf = jnp.zeros(len(Y))
        for i in range(self.k):
            pdf += self.weights[i] * jax.scipy.stats.multivariate_normal.pdf(Y, self.means[i], self.covariances[i])
        return pdf

    def score(self, Y):
        """
        Score function: ∇_x log p(x)
        - Single Gaussian: - (x-μ) Σ^{-1}
        - MoG: sum_i r_i(x) * ( - (x-μ_i) Σ_i^{-1} )
        Returns:
        - (n, d) if Y is (n, d)
        - (d,) if Y is (d,)
        """
        Y_was_1d = (Y.ndim == 1)
        if Y_was_1d:
            Y = Y[None, :]

        logw = jnp.log(self.weights + 1e-32)  # avoid log(0)
        d = self.d
        log2pi = jnp.log(2.0 * jnp.pi)

        def per_component(mu, Sigma, lw):
            # invSigma: (d,d)
            invSigma = jnp.linalg.inv(Sigma)
            diff = Y - mu  # (n,d)
            quad = jnp.einsum('nd,dd,nd->n', diff, invSigma, diff)  # (n,)
            sign, logdet = jnp.linalg.slogdet(Sigma)
            # 若Sigma数值问题导致sign<=0，会让logdet异常；一般Sigma应SPD
            logpdf = -0.5 * (quad + d * log2pi + logdet)  # (n,)

            # Gaussian score: ∇_x log N = -(x-μ) Σ^{-1}
            score_i = -diff @ invSigma  # (n,d)
            return lw + logpdf, score_i

        logterms, scores = jax.vmap(per_component, in_axes=(0, 0, 0))(
            self.means, self.covariances, logw
        )
        # logterms: (k,n), scores: (k,n,d)

        logp = jsp.special.logsumexp(logterms, axis=0)          # (n,)
        resp = jnp.exp(logterms - logp[None, :])               # (k,n)

        score_mix = jnp.einsum('kn,knd->nd', resp, scores)     # (n,d)

        return score_mix.squeeze(0) if Y_was_1d else score_mix

    # def score(self, Y):
    #     def log_pdf(y):
    #         return jnp.log(self.pdf(y)).sum()
    #     score_fn = jax.grad(log_pdf)
    #     return score_fn(Y)
    
    def integral(self):
        if self.integrand_name == 'square':
            integral = 0
            for i in range(self.k):
                integral += self.weights[i] * (jnp.trace(self.covariances[i, :, :]) + jnp.linalg.norm(self.means[i])**2)
        elif self.integrand_name == 'neg_exp':
            integral = 0
            for i in range(self.k):
        #         cov_inv = jnp.linalg.inv(self.covariances[i, :, :])
        #         temp = jnp.exp(0.5 * (self.means[i].T @ cov_inv @ jnp.linalg.inv(cov_inv + 2 * jnp.eye(self.d)) @ cov_inv @ self.means[i]))
        #         temp *= jnp.exp(-0.5 * self.means[i].T @ cov_inv @ self.means[i])
        #         temp *= jnp.sqrt(jnp.linalg.det(2 * self.covariances[i, :, :] + jnp.eye(self.d)))
        #         cov_new = jnp.linalg.inv(cov_inv + 2 * jnp.eye(self.d))
        #         integral += self.weights[i] * temp * jnp.sqrt(jnp.linalg.det(cov_inv)) * jnp.sqrt(jnp.linalg.det(cov_new))
                mu = self.means[i]
                Sigma = self.covariances[i]
                Sigma_inv = jnp.linalg.inv(Sigma)
                A = (2 / (self.d ** 2 / 2)) * jnp.eye(self.d) + Sigma_inv
                A_inv = jnp.linalg.inv(A)

                exponent = 0.5 * mu.T @ Sigma_inv @ A_inv @ Sigma_inv @ mu - 0.5 * mu.T @ Sigma_inv @ mu
                det_term = jnp.sqrt(jnp.linalg.det(A_inv)) / jnp.sqrt(jnp.linalg.det(Sigma))

                temp = jnp.exp(exponent) * det_term
                integral += self.weights[i] * temp
        return integral
    
class LaplacianDistribution:
    def __init__(
        self,
        kernel,
        loc,
        scale,
        integrand_name,
        mc_kme_samples=4096,
        rng_key=jax.random.PRNGKey(0),
    ):
        """
        Independent multivariate Laplace distribution:
            X_j ~ Laplace(loc_j, scale_j), independently for each dimension.

        Parameters
        ----------
        kernel : kernel object
        loc : (d,) array
            Location parameter.
        scale : float or (d,) array
            Positive scale(s).
        integrand_name : str
            'square' or 'neg_exp'
        mc_kme_samples : int
            Number of MC samples used to approximate non-Stein kernel mean embeddings.
        rng_key : jax.random.PRNGKey
            Used to build the MC cache for mean_embedding / mean_mean_embedding.
        """
        self.kernel = kernel
        self.loc = jnp.asarray(loc)
        self.d = self.loc.shape[0]

        self.scale = jnp.asarray(scale)
        if self.scale.ndim == 0:
            self.scale = jnp.ones(self.d) * self.scale
        assert self.scale.shape == (self.d,), "scale must be scalar or shape (d,)"
        assert jnp.all(self.scale > 0), "scale must be positive"

        self.integrand_name = integrand_name
        if integrand_name == 'square':
            self.integrand = lambda x: (x**2).sum(1)
        elif integrand_name == 'neg_exp':
            self.integrand = lambda x: jnp.exp(-(x**2).sum(1) / (self.d ** 2 / 2))
        else:
            raise ValueError('Function not recognized!')

        # For non-Stein kernels, we approximate KME by MC once and cache it.
        self.mc_kme_samples = mc_kme_samples
        self._mc_key = rng_key
        self._mc_samples = self.sample(mc_kme_samples, rng_key)

        if not getattr(self.kernel, "is_stein", False):
            block = 1024
            total, count = 0.0, 0
            for i in range(0, mc_kme_samples, block):
                Xi = self._mc_samples[i:i+block]
                for j in range(0, mc_kme_samples, block):
                    Xj = self._mc_samples[j:j+block]
                    D = self.kernel.make_distance_matrix(Xi, Xj)
                    total += D.sum()
                    count += D.size
            self.double_kme = total / count
        else:
            self.double_kme = jnp.array(0.0)

    def mean_embedding(self, Y):
        # Stein kernel case: E_{X~p}[k_p(X, y)] = 0
        if getattr(self.kernel, "is_stein", False):
            if Y.ndim == 1:
                return jnp.array(0.0)
            else:
                return jnp.zeros(Y.shape[:-1])

        # MC approximation using cached samples
        if Y.ndim == 1:
            Y = Y[None, :]
            block = 1024
            total, count = 0.0, 0
            for i in range(0, self.mc_kme_samples, block):
                Xi = self._mc_samples[i:i+block]
                D = self.kernel.make_distance_matrix(Y, Xi).sum(1)
                total += D
                count += Xi.shape[0]
            return (total / count).squeeze()
        elif Y.ndim == 2:
            block = 1024
            total, count = 0.0, 0
            for i in range(0, self.mc_kme_samples, block):
                Xi = self._mc_samples[i:i+block]
                D = self.kernel.make_distance_matrix(Y, Xi).sum(1)
                total += D
                count += Xi.shape[0]
            return total / count
        elif Y.ndim == 3:
            d = Y.shape[-1]
            Y2 = Y.reshape((-1, d))
            block = 1024
            total, count = 0.0, 0
            for i in range(0, self.mc_kme_samples, block):
                Xi = self._mc_samples[i:i+block]
                D = self.kernel.make_distance_matrix(Y2, Xi).sum(1)
                total += D
                count += Xi.shape[0]
            kme2 = total / count
            return kme2.reshape(Y.shape[:-1])
        else:
            raise ValueError("Y must have ndim in {1,2,3}")

    def mean_mean_embedding(self):
        if getattr(self.kernel, "is_stein", False):
            return jnp.array(0.0)
        return self.double_kme

    def sample(self, sample_size, rng_key):
        """
        Sample i.i.d from the independent multivariate Laplace distribution.
        """
        z = jax.random.laplace(rng_key, shape=(sample_size, self.d))
        return self.loc[None, :] + z * self.scale[None, :]

    def qmc_sample(self, sample_size, rng_key):
        """
        QMC sample using Sobol + inverse CDF of 1D Laplace, dimension-wise.
        """
        # generate a python int seed from JAX key
        seed = int(jax.random.randint(rng_key, shape=(), minval=0, maxval=2**31-1))
        sobol = scipy.stats.qmc.Sobol(d=self.d, scramble=True, seed=seed)
        u = jnp.array(sobol.random(sample_size))

        # avoid exactly 0 or 1
        eps = 1e-12
        u = jnp.clip(u, eps, 1.0 - eps)

        # scipy's laplace.ppf supports vectorized loc/scale
        samples = scipy.stats.laplace.ppf(
            np.array(u),
            loc=np.array(self.loc),
            scale=np.array(self.scale),
        )
        return jnp.array(samples)

    def pdf(self, Y):
        """
        Product Laplace pdf:
            p(x) = prod_j [1/(2 b_j) * exp(-|x_j-loc_j|/b_j)]
        """
        Y_was_1d = (Y.ndim == 1)
        if Y_was_1d:
            Y = Y[None, :]
        logpdf = -jnp.sum(jnp.abs(Y - self.loc[None, :]) / self.scale[None, :], axis=1)
        logpdf -= jnp.sum(jnp.log(2.0 * self.scale))
        out = jnp.exp(logpdf)
        return out.squeeze(0) if Y_was_1d else out

    def score(self, Y):
        """
        Score function of independent Laplace:
            ∇_x log p(x) = -sign(x-loc)/scale
        At x=loc, subgradient is not unique; we return 0 there.
        """
        Y_was_1d = (Y.ndim == 1)
        if Y_was_1d:
            Y = Y[None, :]

        diff = Y - self.loc[None, :]
        sgn = jnp.sign(diff)  # sign(0)=0
        score = -sgn / self.scale[None, :]
        return score.squeeze(0) if Y_was_1d else score

    def integral(self):
        """
        Compute E[f(X)].

        - square:
            E[||X||^2] = sum_j (loc_j^2 + 2 scale_j^2)

        - neg_exp:
            No simple closed form implemented here; use MC approximation.
        """
        if self.integrand_name == 'square':
            return jnp.sum(self.loc**2 + 2.0 * self.scale**2)
        elif self.integrand_name == 'neg_exp':
            return self.integrand(self._mc_samples).mean()
        else:
            raise ValueError('Function not recognized!')

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from tqdm import tqdm


class RadialLaplacianDistribution:
    def __init__(
        self,
        kernel,
        loc,
        scale,
        integrand_name,
        mc_kme_samples=4096,
        rng_key=jax.random.PRNGKey(0),
    ):
        """
        d-dimensional isotropic radial distribution:
            p(x) ∝ exp(-||x-loc||_2 / scale)

        Parameters
        ----------
        loc : (d,) array
        scale : positive scalar
        """
        self.kernel = kernel
        self.loc = jnp.asarray(loc)
        self.d = self.loc.shape[0]
        self.scale = jnp.asarray(scale)

        assert self.scale.ndim == 0, "scale should be a scalar for radial Laplacian"
        assert self.scale > 0, "scale must be positive"

        self.integrand_name = integrand_name
        if integrand_name == 'square':
            self.integrand = lambda x: (x**2).sum(1)
        elif integrand_name == 'neg_exp':
            self.integrand = lambda x: jnp.exp(-(x**2).sum(1) / (self.d ** 2 / 2))
        else:
            raise ValueError('Function not recognized!')

        self.mc_kme_samples = mc_kme_samples
        self._mc_samples = self.sample(mc_kme_samples, rng_key)

        if not getattr(self.kernel, "is_stein", False):
            block = 1024
            total, count = 0.0, 0
            for i in range(0, mc_kme_samples, block):
                Xi = self._mc_samples[i:i+block]
                for j in range(0, mc_kme_samples, block):
                    Xj = self._mc_samples[j:j+block]
                    D = self.kernel.make_distance_matrix(Xi, Xj)
                    total += D.sum()
                    count += D.size
            self.double_kme = total / count
        else:
            self.double_kme = jnp.array(0.0)

    def mean_embedding(self, Y):
        if getattr(self.kernel, "is_stein", False):
            if Y.ndim == 1:
                return jnp.array(0.0)
            return jnp.zeros(Y.shape[:-1])

        if Y.ndim == 1:
            Y = Y[None, :]
            block = 1024
            total, count = 0.0, 0
            for i in range(0, self.mc_kme_samples, block):
                Xi = self._mc_samples[i:i+block]
                D = self.kernel.make_distance_matrix(Y, Xi).sum(1)
                total += D
                count += Xi.shape[0]
            return (total / count).squeeze()

        elif Y.ndim == 2:
            block = 1024
            total, count = 0.0, 0
            for i in range(0, self.mc_kme_samples, block):
                Xi = self._mc_samples[i:i+block]
                D = self.kernel.make_distance_matrix(Y, Xi).sum(1)
                total += D
                count += Xi.shape[0]
            return total / count

        elif Y.ndim == 3:
            d = Y.shape[-1]
            Y2 = Y.reshape((-1, d))
            block = 1024
            total, count = 0.0, 0
            for i in range(0, self.mc_kme_samples, block):
                Xi = self._mc_samples[i:i+block]
                D = self.kernel.make_distance_matrix(Y2, Xi).sum(1)
                total += D
                count += Xi.shape[0]
            return (total / count).reshape(Y.shape[:-1])

        else:
            raise ValueError("Y must have ndim in {1,2,3}")

    def mean_mean_embedding(self):
        if getattr(self.kernel, "is_stein", False):
            return jnp.array(0.0)
        return self.double_kme

    def sample(self, sample_size, rng_key):
        """
        Sample from p(x) ∝ exp(-||x-loc||/scale).
        Use:
            direction ~ uniform on sphere
            radius ~ Gamma(shape=d, scale=scale)
        """
        key_dir, key_rad = jax.random.split(rng_key)

        g = jax.random.normal(key_dir, shape=(sample_size, self.d))
        g_norm = jnp.linalg.norm(g, axis=1, keepdims=True)
        direction = g / (g_norm + 1e-12)

        radius = jax.random.gamma(key_rad, a=self.d, shape=(sample_size,)) * self.scale

        return self.loc[None, :] + direction * radius[:, None]

    def pdf(self, Y):
        """
        p(x) = c_d / scale^d * exp(-||x-loc||/scale)
        where
            c_d = Gamma(d/2) / (2 * pi^(d/2) * Gamma(d))
        """
        Y_was_1d = (Y.ndim == 1)
        if Y_was_1d:
            Y = Y[None, :]

        r = jnp.linalg.norm(Y - self.loc[None, :], axis=1)

        log_cd = (
            jsp.special.gammaln(self.d / 2.0)
            - jnp.log(2.0)
            - (self.d / 2.0) * jnp.log(jnp.pi)
            - jsp.special.gammaln(self.d)
        )
        logpdf = log_cd - self.d * jnp.log(self.scale) - r / self.scale
        out = jnp.exp(logpdf)
        return out.squeeze(0) if Y_was_1d else out

    def score(self, Y):
        """
        ∇ log p(x) = -(x-loc)/(scale * ||x-loc||)
        Use eps for numerical stability at x=loc.
        """
        Y_was_1d = (Y.ndim == 1)
        if Y_was_1d:
            Y = Y[None, :]

        diff = Y - self.loc[None, :]
        r = jnp.linalg.norm(diff, axis=1, keepdims=True)
        score = -diff / (self.scale * (r + 1e-12))
        return score.squeeze(0) if Y_was_1d else score

    def integral(self):
        if self.integrand_name == 'square':
            # E ||X||^2 = ||loc||^2 + E[R^2] + 2 loc·E[RU]
            # E[U]=0 so cross term vanishes.
            # R ~ Gamma(shape=d, scale=scale)
            # E[R^2] = Var(R) + E[R]^2 = d s^2 + d^2 s^2 = d(d+1)s^2
            return jnp.sum(self.loc**2) + self.d * (self.d + 1) * self.scale**2

        elif self.integrand_name == 'neg_exp':
            return self.integrand(self._mc_samples).mean()

        else:
            raise ValueError('Function not recognized!')

class Empirical_Distribution:
    def __init__(self, kernel, samples, integrand_name):
        self.kernel = kernel
        self.samples = samples
        self.integrand_name = integrand_name
        self.n = len(samples)
        self.d = samples.shape[1]
        
        if integrand_name == 'square':
            self.integrand = lambda x: (x**2).sum(1)
        elif integrand_name == 'neg_exp':
            self.integrand = lambda x: jnp.exp(-(x**2).sum(1))
        else:
            raise ValueError('Function not recognized!')
        # Compute the double KME once during initialization
        # Because samples are fixed
        # Also because it is very memory intensive to compute repeatedly
        block = 1024
        total, count = 0.0, 0
        for i in tqdm(range(0, self.n, block)):
            Xi = samples[i:i+block]

            for j in range(0, self.n, block):
                Xj = samples[j:j+block]
                D = kernel.make_distance_matrix(Xi, Xj)  # (bi, bj)

                total += D.sum()
                count += D.size
        self.double_kme = total / count

    def mean_embedding(self, Y):
        """
        Compute the kernel mean embedding.

        Parameters:
        - Y: (n, d) array of points to evaluate 

        Returns:
        - pdf: (n,) array of PDF values.
        """
        if Y.ndim == 1:
            Y = Y[None, :]
            block = 1024
            total, count = 0.0, 0
            for i in range(0, self.n, block):
                Xi = self.samples[i:i+block]
                D = self.kernel.make_distance_matrix(Y, Xi).sum(1)
                total += D
                count += Xi.shape[0]
            kme = (total / count).squeeze()
        elif Y.ndim == 2:
            block = 1024
            total, count = 0.0, 0
            for i in range(0, self.n, block):
                Xi = self.samples[i:i+block]
                D = self.kernel.make_distance_matrix(Y, Xi).sum(1)
                total += D
                count += Xi.shape[0]
            kme = total / count
        elif Y.ndim == 3:
            d = Y.shape[-1]
            Y2 = Y.reshape((-1, d))
            block = 1024
            total, count = 0.0, 0
            for i in range(0, self.n, block):
                Xi = self.samples[i:i+block]
                D = self.kernel.make_distance_matrix(Y2, Xi).sum(1)
                total += D
                count += Xi.shape[0]
            kme2 = total / count
            kme = kme2.reshape(Y.shape[:-1]) 
        return kme
    
    def mean_mean_embedding(self):
        """
        Compute the kernel mean embedding of the empirical distribution.

        Returns:
        - double_kme: scalar, the value of the double integral.
        """
        # double_kme = self.kernel.make_distance_matrix(self.samples, self.samples).mean()
        return self.double_kme
    
    def sample(self, sample_size, rng_key):
        """
        Sample i.i.d from the empirical distribution.

        Parameters:
        - sample_size: int, the number of samples to draw.
        - rng_key: JAX PRNGKey for reproducibility.

        Returns:
        - samples: (sample_size, d) array of samples.
        """
        rng_key, _ = jax.random.split(rng_key)
        indices = jax.random.choice(rng_key, self.n, shape=(sample_size,), replace=True)
        return self.samples[indices]
    
    def integral(self):
        """
        Compute the integral of the empirical distribution.

        Returns:
        - integral: scalar, the value of the integral.
        """
        if self.integrand_name == 'square':
            integral = (self.samples**2).sum(1).mean()
        elif self.integrand_name == 'neg_exp':
            integral = jnp.exp(-(self.samples**2).sum(1)).mean()
        return integral
    

class Cross:
    def __init__(self, kernel, w, h, k, skip):
        """
        A class that takes cross distribution.

        Parameters:
        - kernel: the kernel
        """
        self.kernel = kernel
        self.w = w
        self.h = h
        self.k = k
        self.skip = skip
        area_overlap = w * w
        area_vertical_only = w * h - area_overlap
        area_horizontal_only = w * h - area_overlap
        self.area_total = (area_vertical_only + area_horizontal_only + area_overlap) * self.k * 2
        self.integrand = lambda x: 0

    def mean_embedding(self, Y):
        final_kme = jnp.zeros(Y.shape[0])
        for i in range(-1, self.k-1, 1):
            kme_1 = self.kernel.mean_embedding_uniform(jnp.array([-self.w/2 + self.skip * i, -self.h/2]), 
                                                       jnp.array([self.w/2 + self.skip * i, self.h/2]), Y)
            kme_1 += self.kernel.mean_embedding_uniform(jnp.array([-self.w/2 + self.skip * i, -self.h/2 + self.skip]), 
                                                       jnp.array([self.w/2 + self.skip * i, self.h/2 + self.skip]), Y)
            
            kme_2 = self.kernel.mean_embedding_uniform(jnp.array([-self.h/2 + self.skip * i, -self.w/2]), 
                                                       jnp.array([-self.w/2 + self.skip * i, self.w/2]), Y)
            kme_2 += self.kernel.mean_embedding_uniform(jnp.array([-self.h/2 + self.skip * i, -self.w/2 + self.skip]),
                                                       jnp.array([-self.w/2 + self.skip * i, self.w/2 + self.skip]), Y)
            
            kme_3 = self.kernel.mean_embedding_uniform(jnp.array([self.w/2 + self.skip * i, -self.w/2]), 
                                                       jnp.array([self.h/2 + self.skip * i, self.w/2]), Y)
            kme_3 += self.kernel.mean_embedding_uniform(jnp.array([self.w/2 + self.skip * i, -self.w/2 + self.skip]),
                                                         jnp.array([self.h/2 + self.skip * i, self.w/2 + self.skip]), Y)
            final_kme += kme_1 * self.w * self.h / self.area_total * 2
            final_kme += kme_2 * (self.w * self.h - self.w * self.w) / 2 / self.area_total * 2
            final_kme += kme_3 * (self.w * self.h - self.w * self.w) / 2 / self.area_total * 2
        return final_kme
    
    def sample(self, sample_size, rng_key):
        """
        Sample i.i.d from the Cross distribution.

        Parameters:
        - sample_size: int, the number of samples to draw.
        - rng_key: JAX PRNGKey for reproducibility.

        Returns:
        - samples: (sample_size, d) array of samples.
        """
        rng_key, _ = jax.random.split(rng_key)
        minval_all = jnp.zeros((3 * self.k * 2, 2))
        maxval_all = jnp.zeros((3 * self.k * 2, 2))
        weights = jnp.zeros((3 * self.k * 2, ))
        for i in range(0, self.k, 1):
            loc = i - 1
            minval_all = minval_all.at[3*i: 3*(i+1), :].set(jnp.array([[-self.w/2 + self.skip * loc, -self.h/2], 
                                [-self.h/2 + self.skip * loc, -self.w/2], 
                                [self.w/2 + self.skip * loc, -self.w/2]]))
            minval_all = minval_all.at[3*(i+self.k): 3*(i+1+self.k), :].set(jnp.array([[-self.w/2 + self.skip * loc, -self.h/2 + self.skip], 
                                [-self.h/2 + self.skip * loc, -self.w/2 + self.skip], 
                                [self.w/2 + self.skip * loc, -self.w/2 + self.skip]]))
            
            maxval_all = maxval_all.at[3*i: 3*(i+1), :].set(jnp.array([[self.w/2 + self.skip * loc, self.h/2], 
                                [-self.w/2 + self.skip * loc, self.w/2],
                                [self.h/2 + self.skip * loc, self.w/2]]))
            maxval_all = maxval_all.at[3*(i+self.k): 3*(i+1+self.k), :].set(jnp.array([[self.w/2 + self.skip * loc, self.h/2 + self.skip], 
                                [-self.w/2 + self.skip * loc, self.w/2 + self.skip],
                                [self.h/2 + self.skip * loc, self.w/2 + self.skip]]))
            
            weights = weights.at[3*i: 3*(i+1)].set(jnp.array([self.w * self.h / self.area_total, 
                             (self.w * self.h - self.w * self.w) / 2 / self.area_total, 
                             (self.w * self.h - self.w * self.w) / 2 / self.area_total]))
            weights = weights.at[3*(i+self.k): 3*(i+1+self.k)].set(jnp.array([self.w * self.h / self.area_total, 
                             (self.w * self.h - self.w * self.w) / 2 / self.area_total, 
                             (self.w * self.h - self.w * self.w) / 2 / self.area_total]))
            
        component_indices = jax.random.choice(rng_key, 3 * self.k * 2, shape=(sample_size,), p=weights)

        minvals = minval_all[component_indices, :]
        maxvals = maxval_all[component_indices, :]

        def sample_uniform(minval, maxval, key):
            return jax.random.uniform(key, shape=(2,), minval=minval, maxval=maxval)

        subkeys = jax.random.split(rng_key, sample_size)
        samples = jax.vmap(sample_uniform)(minvals, maxvals, subkeys)
        return samples

    def pdf(self, Y):
        """
        Compute the probability density function of the Cross distribution.
        It is essentially a uniform distribution over the cross shape.

        Parameters:
        - Y: (n, d) array of points to evaluate the PDF at.

        Returns:
        - pdf: (n,) array of PDF values.
        """
        x, y = Y[:, 0], Y[:, 1]
        in_cross_all = jnp.zeros((Y.shape[0], self.k))
        for i in range(-1, self.k-1, 1):
            in_vertical = (jnp.abs(x - self.skip * i) <= self.w / 2) & (jnp.abs(y) <= self.h / 2)
            in_horizontal = (jnp.abs(x - self.skip * i) <= self.h / 2) & (jnp.abs(y) <= self.w / 2)

            in_vertical_ = (jnp.abs(x - self.skip * i) <= self.w / 2) & (jnp.abs(y - self.skip) <= self.h / 2)
            in_horizontal_ = (jnp.abs(x - self.skip * i) <= self.h / 2) & (jnp.abs(y - self.skip) <= self.w / 2)
            in_cross_all = in_cross_all.at[:, i].set(in_vertical | in_horizontal | in_vertical_ | in_horizontal_)

        pdf_values = jnp.where(jnp.any(in_cross_all, axis=1), 1.0, jnp.nan)
        return pdf_values
    
    def integral(self):
        return 0.0
    
