from dataclasses import dataclass
from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp
import numpy
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalDiag
from tensorflow_probability.substrates.jax import mcmc


def make_ln_prob(
        type: Literal['total', 'ordinal'],
        x: jnp.ndarray,
        *,
        sigma2: Optional[jnp.ndarray],
        with_dispersion: bool,
        internal_params_to_ls: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        residual_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        ln_prior: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
):
    """Make ln probability function for ordinal or total least squares

    x : jnp.ndarray
        2D array of shape (n_samples, n_vars)
    type : {'total', 'ordinal'}
        Whether to use total or ordinal least squares
    sigma2 : jnp.ndarray or None
        2D array of shape (n_samples, n_vars) of variances of each variable
    with_dispersion : bool
        Whether to include a dispersion parameter
    internal_params_to_ls : Callable or None
        Function to convert internal parameters to least squares parameters.
    residual_fn : Callable or None
        Function to transform (non-squared) residuals divided by total
        dispersion.
    ln_prior : Callable or None
        Function to compute natural logarithm of prior probability of
        internal parameters.
    """
    if sigma2 is None and not with_dispersion:
        raise ValueError('Must have at least one of sigma2 or with_dispersion')

    intercept_idx = -2 if with_dispersion else -1

    match type:
        case 'total':
            total_ls = True
        case 'ordinal':
            total_ls = False
        case _:
            raise ValueError(f'Unknown type {type}')

    # segfault
    # @jax.jit
    def ln_prob(params):
        if ln_prior is not None:
            prior = ln_prior(params)
        else:
            prior = 0.0
        if internal_params_to_ls is not None:
            params = internal_params_to_ls(params)

        slopes_but_last = params[:intercept_idx]
        slopes = jnp.append(slopes_but_last, -1.0)
        intercept = params[intercept_idx]
        if total_ls:
            slopes_norm = jnp.linalg.norm(slopes)
            slopes /= slopes_norm
            intercept /= slopes_norm
        if with_dispersion:
            dispersion = params[-1]
        else:
            dispersion = 0.0
        del params

        residuals = jnp.dot(x, slopes) + intercept
        if sigma2 is not None:
            sigma2_total = jnp.dot(sigma2, slopes**2) + dispersion**2
        else:
            sigma2_total = jnp.full(x.shape[0], dispersion**2)

        # We could merge these two branches, but it is much slower to use standard normal distribution
        if residual_fn is not None:
            residuals = residual_fn(residuals / sigma2_total)
            model = MultivariateNormalDiag(loc=0.0, scale_diag=jnp.ones(x.shape[0])).log_prob(residuals)
        else:
            model = MultivariateNormalDiag(loc=0.0, scale_diag=jnp.sqrt(sigma2_total)).log_prob(residuals)

        return model + prior

    return ln_prob


DEFAULT_NUTS_STEP_SIZE = 1e-4


def least_squares(
        type: Literal['total', 'ordinal'],
        x,
        *,
        sigma2=None,
        with_dispersion: bool = True,
        initial_slopes: Optional = None,
        initial_intercept: Optional[float] = 0.0,
        initial_dispersion: Optional[float] = 1.0,
        num_samples: int = 1_000,
        num_burnin: Optional[int] = None,
        random_seed: int = 0,
        ls_params_to_internal: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        internal_params_to_ls: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        residual_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        ln_prior: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        nuts_kwargs: Optional[dict] = None,
):
    f"""Linear least squares regression

    Linear regression model with dispersion and known observation errors,
    ordinal or total least squares.
    Note that for both types of least squares, the last slope is fixed to -1.

    Parameters
    ----------
    x : array_like
        2D array of shape (n_obs, n_vars) with observations
    type : {'total', 'ordinal'}
        Type of least squares to solve. If 'ordinal', the target (y) is given
        by the last column of `x` and `sigma2`. If 'total', the target is zero.
    sigma2 : array_like, optional
        2D array of shape (n_obs, n_vars) with the variance of each
        observation. If not given assumed no observation errors. One of sigma2
        and with_dispersion must be given.
    with_dispersion : bool, optional
        If `True`, include a dispersion parameter in the model. If `False`
        assume no dispersion. One of sigma2 and with_dispersion must be given.
    initial_slopes : array_like, optional:
        Initial values for the slopes. Shape must be (n_vars,), default is 
        [1.0, 0.0, ..., 0.0, -1.0].
    initial_intercept : float, optional
        Initial value for the intercept.
    initial_dispersion : float, optional
        Initial value for the dispersion.
    num_samples : int, optional
        Number of samples to draw from the posterior.
    num_burnin : int, optional
        Number of samples to discard as burn-in. If not given, use 10% of
        `num_samples`.
    random_seed : int, optional
        Random seed to use for the MCMC chain.
    ls_params_to_internal, internal_params_to_ls : callable, optional
        Functions to convert between the parameters of the linear least squares
        [slopes_1, ..., slopes_n_var, intercept, dispersion] and some internal
        representation. The internal representation is used for the MCMC chain.
        Both functions must take a 1D jax array as input and return
        a 1D jax-array using jax only.
    residual_fn : callable, optional
        Function to apply to (non-squared) residuals divided by the total
        dispersion before computing the standard normal likelihood.
        The function takes a jax array and returns a jax array with the same
        shape. It must be written in jax. Default is to use the identity
        function.
        Note that for some reason usage of this function affects performance
        dramatically.
    ln_prior : callable, optional
        Function to compute the natural logarithm of prior. The function takes
        a jax array and returns a jax array with the same shape. It must be
        written in jax.
    nuts_kwargs: dict, optional
        Keyword arguments to pass to
        `tensorflow_probability.mcmc.NoUTurnSampler`. Default is to use
        `step_size={DEFAULT_NUTS_STEP_SIZE}`, which also will be added to
        the dict if no `step_size` is given.
    """
    x = jnp.asarray(x)
    if x.ndim != 2:
        raise ValueError('x must be a 2D array')
    _n_obs, n_vars = x.shape
    if n_vars < 2:
        raise ValueError('x must have at least two columns')

    if sigma2 is not None:
        sigma2 = jnp.asarray(sigma2)
        if x.shape != sigma2.shape:
            raise ValueError('x and sigma2 must have the same shape')

    # We need (n_vars-1) slopes only, while the last one is fixed to -1
    if initial_slopes is None:
        initial_slopes = jnp.r_[1.0, jnp.zeros(n_vars - 2)]
    else:
        initial_slopes = jnp.asarray(initial_slopes)
        if initial_slopes.ndim != 1:
            raise ValueError('initial_slopes must be a 1D array')
        if initial_slopes.shape[0] != n_vars:
            raise ValueError('initial_slopes must have the same length as x for total least squares')
        slope_norm = -initial_slopes[-1]
        initial_slopes = initial_slopes[:-1] / slope_norm
        initial_intercept /= slope_norm

    if num_burnin is None:
        num_burnin = num_samples // 10

    key = jax.random.PRNGKey(random_seed)

    if (ls_params_to_internal is None) ^ (internal_params_to_ls is None):
        raise ValueError('Both ls_params_to_internal and internal_params_to_ls must be given')

    if nuts_kwargs is None:
        nuts_kwargs = {}
    nuts_kwargs = dict(step_size=DEFAULT_NUTS_STEP_SIZE) | nuts_kwargs

    initial_state = jnp.r_[
        initial_slopes,
        initial_intercept,
        initial_dispersion,
    ]
    if not with_dispersion:
        initial_state = initial_state[:-1]
    if ls_params_to_internal is not None:
        initial_state = ls_params_to_internal(initial_state)

    ln_prob = make_ln_prob(
        type=type,
        x=x,
        sigma2=sigma2,
        with_dispersion=with_dispersion,
        internal_params_to_ls=internal_params_to_ls,
        residual_fn=residual_fn,
        ln_prior=ln_prior,
    )

    # segfault
    # @jax.jit
    def run_chain(*, key, state):
        kernel = mcmc.NoUTurnSampler(ln_prob, **nuts_kwargs)
        result = mcmc.sample_chain(
            num_samples,
            num_burnin_steps=num_burnin,
            current_state=state,
            kernel=kernel,
            return_final_kernel_results=True,
            trace_fn=lambda _, results: results.target_log_prob,
            seed=key,
        )
        return result.all_states, result.trace, result.final_kernel_results

    key, subkey = jax.random.split(key)
    states, ln_probs, _ = run_chain(key=subkey, state=initial_state, ); del subkey

    return states, ln_probs


@dataclass(slots=True, kw_only=True)
class ChainStats:
    """Statistics of a chain of samples"""
    mean: numpy.ndarray
    median: numpy.ndarray
    variance: numpy.ndarray
    standard_error: numpy.ndarray
    effective_sample_size: numpy.ndarray
    rhat: numpy.ndarray
    percentiles: numpy.ndarray

    def __post_init__(self):
        for name in self.__slots__:
            setattr(self, name, numpy.asarray(getattr(self, name)))

    @classmethod
    def from_states(cls, states):
        states = jnp.asarray(states)
        effective_sample_size = mcmc.diagnostic.effective_sample_size(states)
        mean = jnp.mean(states, axis=0)
        median = jnp.median(states, axis=0)
        variance = jnp.var(states, axis=0)
        standard_error = jnp.sqrt(variance / effective_sample_size)
        percentiles = jnp.percentile(states, jnp.arange(1, 100), axis=0)
        rhat = mcmc.diagnostic.potential_scale_reduction(states)
        return cls(
            mean=mean,
            median=median,
            variance=variance,
            standard_error=standard_error,
            effective_sample_size=effective_sample_size,
            percentiles=percentiles,
            rhat=rhat
        )
