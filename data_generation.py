#
# data_generation.py
#
# Project: Detecting Stable States in Non-Stationary Time Series
# Date: 2026-07-27
# Author: Florian Heinrichs
#
# Script to generate data.

from typing import Literal

import numpy as np

SignalMode = Literal["lvl", "drv"]
Calibration = Literal["gumbel", "gaussian"]

def generate_data(n: int, n_samples: int, mean: str, delta: float,
                  error: str, sigma: str, std: float = 1.0) -> np.ndarray:
    """
    Wrapper function to generate noisy data.

    :param n: Length of generated time series.
    :param n_samples: Number of generated time series.
    :param mean: Type of mean in ['1', '2', 'abrupt', 'const'].
    :param error: Type of error in ['iid', 'ar', 'ma'].
    :param sigma: Type of sigma in ['0', '1', '2', '3'].
    :param std: Standard deviation of i.i.d. innovations.
    :return: NumPy array containing noisy data.
    """
    _check_delta(delta)

    if mean == 'drv_settling':
        mean = mu_deriv_settling(n, delta=delta)
    elif mean == 'drv_settling_drift':
        mean = mu_deriv_settling_drift(n, delta=delta)
    elif mean == 'drv_plateau':
        mean = mu_deriv_true_plateau(n, delta=delta)
    elif mean == 'drv_tmp_plateau':
        mean = mu_deriv_temporary_plateau_drift(n, delta=delta)
    elif mean == 'lvl_linear':
        mean = mu_level_overall_mean(n, delta=delta)
    elif mean == 'lvl_bump':
        mean = mu_level_fixed_baseline_bump(n, delta=delta)
    else:
        raise ValueError("Mean type unknown.")

    if error == 'iid':
        error = generate_iid(n, n_samples, std=std)
    elif error == 'ar':
        error = generate_ar(n, n_samples, std=std)
    elif error == 'ma':
        error = generate_ma(n, n_samples, std=std)
    elif error == 'loc_stat':
        error = generate_loc_stat(n, n_samples, std=std)
    else:
        raise ValueError("Error type unknown.")

    if sigma == '0':
        sigma = np.ones((n, 1)) / 2
    elif sigma == '1':
        sigma = sigma_1(n)
    elif sigma == '2':
        sigma = sigma_2(n)
    elif sigma == '3':
        sigma = sigma_3(n)
    else:
        raise ValueError("Sigma type unknown.")

    return mean + sigma * error


def _check_delta(delta: float) -> None:
    if not (0 <= delta <= 1):
        raise ValueError("delta must be in [0, 1].")


def _smootherstep(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, 0.0, 1.0)
    return 10 * z**3 - 15 * z**4 + 6 * z**5


def _smootherstep_deriv(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, 0.0, 1.0)
    d = 30 * z**2 * (1 - z)**2
    d[(x <= 0) | (x >= 1)] = 0.0
    return d


def _dinf_grid(x: np.ndarray, d: np.ndarray, delta: float) -> float:
    """
    Vectorized grid approximation of

        inf_t sup_{s in [t, t + delta]} |d(s)|.

    Assumes x is an increasing, approximately equidistant grid on [0, 1].
    """
    d_abs = np.abs(d.reshape(-1))
    n = len(d_abs)

    if delta == 0:
        return float(np.min(d_abs))

    # Number of grid points in a window of length approximately delta.
    #
    # If x = np.linspace(0, 1, n), then spacing is 1 / (n - 1).
    # The window [t, t + delta] contains approximately floor(delta / h) + 1 points.
    h = x[1] - x[0]
    w = int(np.floor(delta / h)) + 1
    w = max(1, min(w, n))

    # Starting indices i with x[i] <= 1 - delta.
    m = np.searchsorted(x, 1 - delta, side="right")
    if m <= 0:
        return float(np.max(d_abs))

    windows = np.lib.stride_tricks.sliding_window_view(d_abs, window_shape=w)

    # Keep only admissible windows starting at x[i] <= 1 - delta.
    # Also ensure we do not request more starts than sliding_window_view provides.
    m = min(m, windows.shape[0])

    return float(np.min(np.max(windows[:m], axis=1)))


def mu_deriv_settling(n: int, delta: float = 0.1, c: float = 0.1,
                      kappa: float = 6.0) -> np.ndarray:
    """
    Fast initial decrease followed by slow convergence.

    Signal:
        d = mu'

    Normalization:
        d_infty(delta) = 1

    Suggested:
        delta in {0.1, 0.2}
        Delta in {0.75, 1.0, 1.25, 1.5}
    """
    x = np.linspace(0, 1, n)
    mu = -c * x - (1 - c) * (1 - np.exp(-kappa * x)) / kappa
    d = -c - (1 - c) * np.exp(-kappa * x)

    dinf = _dinf_grid(x, d, delta)
    mu = mu / dinf

    return np.expand_dims(mu, axis=1)


def mu_deriv_settling_drift(n: int, delta: float = 0.1, beta: float = 0.15,
                            kappa: float = 6.0) -> np.ndarray:
    """
    Fast decrease, near-stabilization, then slow upward drift.

    Signal:
        d = mu'

    Normalization:
        d_infty(delta) = 1

    Suggested:
        delta in {0.1, 0.2}
        Delta in {0.75, 1.0, 1.25, 1.5}
    """
    x = np.linspace(0, 1, n)
    mu = beta * x + (np.exp(-kappa * x) - 1) / kappa
    d = beta - np.exp(-kappa * x)

    dinf = _dinf_grid(x, d, delta)
    mu = mu / dinf

    return np.expand_dims(mu, axis=1)


def mu_deriv_true_plateau(n: int, delta: float = 0.1,
                          a: float = 0.35) -> np.ndarray:
    """
    Smooth decrease followed by an exactly constant tail.

    Signal:
        d = mu'

    Property:
        d_infty(delta) = 0 if delta <= 1 - a

    Suggested:
        delta in {0.1, 0.2}
        Delta in {0.05, 0.1, 0.25, 0.5}
    """
    x = np.linspace(0, 1, n)
    mu = 1 - _smootherstep(x / a)

    return np.expand_dims(mu, axis=1)


def mu_deriv_temporary_plateau_drift(n: int, delta: float = 0.1,
                                     a: float = 0.30, b: float = 0.55,
                                     gamma: float = 0.20) -> np.ndarray:
    """
    Smooth decrease, temporary exact plateau, then upward drift.

    Signal:
        d = mu'

    Property:
        d_infty(delta) = 0 if delta <= b - a

    Suggested:
        delta in {0.1, 0.2}
        Delta in {0.05, 0.1, 0.25, 0.5}
    """
    x = np.linspace(0, 1, n)
    mu = 1 - _smootherstep(x / a) + gamma * _smootherstep((x - b) / (1 - b))

    return np.expand_dims(mu, axis=1)


def mu_level_overall_mean(n: int, delta: float = 0.1) -> np.ndarray:
    """
    Linear function centered around its overall mean.

    Signal:
        d(t) = mu(t) - g(mu)

    Reference:
        g(mu) = overall mean of mu

    Normalization:
        d_infty(delta) = 1

    Suggested:
        delta in {0.1, 0.2}
        Delta in {0.75, 1.0, 1.25, 1.5}
    """
    x = np.linspace(0, 1, n)

    mu = x - np.mean(x)
    g = np.mean(mu) * np.ones_like(mu)
    d = mu - g

    dinf = _dinf_grid(x, d, delta)
    mu = mu / dinf

    return np.expand_dims(mu, axis=1)


def mu_level_fixed_baseline_bump(n: int,
                                 delta: float = 0.1,
                                 a: float = 0.25,
                                 m: float = 0.50,
                                 b: float = 0.75,
                                 height: float = 1.0) -> np.ndarray:
    """
    Temporary excursion away from a fixed zero baseline.

    Signal:
        d(t) = mu(t) - g(mu, t)

    Reference:
        g(mu, t) = 0

    Property:
        d_infty(delta) = 0 if delta <= max(a, 1 - b)

    Suggested:
        delta in {0.1, 0.2}
        Delta in {0.05, 0.1, 0.25, 0.5}
    """
    x = np.linspace(0, 1, n)
    rise = _smootherstep((x - a) / (m - a))
    fall = 1 - _smootherstep((x - m) / (b - m))

    mu = height * rise * fall

    return np.expand_dims(mu, axis=1)


def generate_iid(n: int, n_samples: int, std: float = 1.0,
                 dist: str = 'normal') -> np.ndarray:
    """
    Generates i.i.d. errors according to the specified distribution.

    :param n: Number of supporting points (observations).
    :param n_samples: Number of (independent) trajectories.
    :param std: Standard deviation of random variables.
    :param dist: Distribution of noise. Defaults to 'normal'.
    :return: Errors at points np.arange(n) / n. Output has shape (n, n_samples).
    """
    rng = np.random.default_rng()

    if dist in ['normal', 'Gaussian']:
        errors = rng.normal(loc=0, scale=1, size=(n, n_samples)) * std
    elif dist == 'uniform':
        c = np.sqrt(3) * std
        errors = rng.uniform(low=-c, high=c, size=(n, n_samples))

    return errors


def generate_ar(n: int, n_samples: int, burn_in: int = 100,
                a: float = 0.5, std: float = 1.0,
                dist: str = 'normal') -> np.ndarray:
    """
    Generates AR(1) errors.

    :param n: Number of supporting points (observations).
    :param n_samples: Number of (independent) trajectories.
    :param burn_in: Time steps used for burn in of AR process.
    :param a: Autoregressive coefficient.
    :param std: Standard deviation of i.i.d. innovations.
    :param dist: Distribution of innovations. Defaults to 'normal'.
    :return: Errors at points np.arange(n) / n. Output has shape (n, n_samples).
    """
    epsilon = generate_iid(n + burn_in, n_samples, std=std, dist=dist)
    errors = np.zeros((n + burn_in, n_samples))

    errors[0] = epsilon[0]
    for i in range(1, n + burn_in):
        errors[i] = a * errors[i - 1] + epsilon[i]

    var_errors = 1 / (1 - a ** 2)
    errors = errors[burn_in:] / np.sqrt(var_errors)

    return errors


def generate_ma(n: int, n_samples: int, a: float = 0.5,
                std: float = 1.0) -> np.ndarray:
    """
    Generates MA(1) errors.

    :param n: Number of supporting points (observations).
    :param n_samples: Number of (independent) trajectories.
    :param a: MA coefficient.
    :param std: Standard deviation of i.i.d. innovations.
    :return: Errors at points np.arange(n) / n. Output has shape (n, n_samples).
    """
    epsilon = generate_iid(n + 1, n_samples, std=std)
    errors = epsilon[1:] + a * epsilon[:-1]

    var_errors = 1 + a ** 2
    errors = errors / np.sqrt(var_errors)

    return errors


def generate_loc_stat(n: int, n_samples: int, std: float = 1.0) -> np.ndarray:
    """
    Generates mixture of AR(1) errors with uniform and normal distribution.

    :param n: Number of supporting points (observations).
    :param n_samples: Number of (independent) trajectories.
    :param std: Standard deviation of i.i.d. innovations.
    :return: Errors at points np.arange(n) / n. Output has shape (n, n_samples).
    """
    eps_normal = generate_ar(n, n_samples, a=0.5, std=std, dist='normal')
    eps_uniform = generate_ar(n, n_samples, a=-0.5, std=std, dist='uniform')

    a = 1 / 2 - np.cos(
        (-np.cos(np.pi * np.linspace(0, 1, n)) + 1) / 2 * np.pi
    )[:, np.newaxis] / 2

    errors = np.sqrt(a) * eps_normal + np.sqrt(1 - a) * eps_uniform

    return errors


def sigma_1(n: int) -> np.ndarray:
    """
    Generates a monotonically increasing function (defined on the unit
    interval).

    :param n: Number of supporting points (observations).
    :return: Function values at points np.arange(n) / n
    """
    x = np.linspace(0, 1, n)
    sigma = 1 / 4 + x / 2

    return np.expand_dims(sigma, axis=1)


def sigma_2(n: int) -> np.ndarray:
    """
    Generates a non-monotonic function (defined on the unit interval).

    :param n: Number of supporting points (observations).
    :return: Function values at points np.arange(n) / n
    """
    x = np.linspace(0, 1, n)
    sigma = 1 / 2 - np.cos(2 * np.pi * x) / 4

    return np.expand_dims(sigma, axis=1)


def sigma_3(n: int) -> np.ndarray:
    """
    Generates a step function (defined on the unit interval).

    :param n: Number of supporting points (observations).
    :return: Function values at points np.arange(n) / n
    """
    sigma = np.ones((n, 1)) / 4
    sigma[n // 2:] = 3 / 4

    return sigma


def display_mu(n: int = 200, include_drv: bool = False):
    import matplotlib.pyplot as plt
    from tueplots import bundles, figsizes

    plt.rcParams.update(bundles.icml2022(family="Times New Roman", usetex=False))
    plt.rcParams.update(figsizes.icml2022_full())

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    x = np.linspace(0, 1, n)

    delta = 0.2
    functions = [mu_deriv_settling, mu_deriv_settling_drift,
                 mu_level_overall_mean, mu_level_fixed_baseline_bump]

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=False,
                             constrained_layout=True)
    axes = axes.ravel()

    for i, func in enumerate(functions):
        ax = axes.flat[i]
        label = f'$\mu_{i+1}$'
        ax.set_title(label)
        ax.plot(x, func(n, delta=delta), color='black', label=label)

        if include_drv and func is mu_deriv_settling:
            c, kappa = 0.1, 6.0
            d_raw = -c - (1 - c) * np.exp(-kappa * x)
            dinf = _dinf_grid(x, d_raw, delta)
            d = d_raw / dinf
            ax.plot(x, d, color="darkgray", ls="dashed", label=f"$\mu_{i+1}'$")
            ax.set_ylim([-2.5, 0.1])
            ax.legend()

        elif include_drv and func is mu_deriv_settling_drift:
            beta, kappa = 0.15, 6.0
            d_raw = beta - np.exp(-kappa * x)
            dinf = _dinf_grid(x, d_raw, delta)
            d = d_raw / dinf
            ax.plot(x, d, color="darkgray", ls="dashed", label=f"$\mu_{i+1}'$")
            ax.set_ylim([-1.5, 2])
            ax.legend()

        ax.tick_params(labelsize=8, width=0.8, length=3)
        ax.grid(False)
        ax.margins(x=0)

    if len(axes.shape) == 2:
        for ax in axes[:3]:
            ax.tick_params(labelbottom=False)

    plt.show()


if __name__ == '__main__':
    display_mu()
    display_mu(include_drv=True)