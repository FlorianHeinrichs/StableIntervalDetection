#
# tests/tests.py
#
# Project: Detecting Stable States in Non-Stationary Time Series
# Date: 2026-07-27
# Author: Florian Heinrichs
#
# Proposed tests based on EVT and Gaussian approximations.


from typing import Literal

import numpy as np
from scipy import signal

from estimators import (bandwidth_cv, estimate_long_run_variance,
                        jackknife_estimator, local_linear_estimator)


SignalMode = Literal["lvl", "drv"]
Calibration = Literal["gumbel", "gaussian", "both"]


def _kernel_eval(x: np.ndarray, mode: str = "quartic") -> np.ndarray:
    """
    Evaluate a compactly supported kernel on [-1, 1].

    :param x: Points at which the kernel is evaluated.
    :param mode: Kernel type, one of 'quartic', 'triweight', 'tricube',
        or 'triangular'.
    :return: Kernel values as NumPy array.
    :raises ValueError: If an unsupported kernel mode is chosen.
    """
    y = np.zeros_like(x, dtype=float)
    m = np.abs(x) <= 1

    if mode == "quartic":
        y[m] = 15 / 16 * (1 - x[m] ** 2) ** 2
    elif mode == "triweight":
        y[m] = 35 / 32 * (1 - x[m] ** 2) ** 3
    elif mode == "tricube":
        y[m] = 70 / 81 * (1 - np.abs(x[m]) ** 3) ** 3
    elif mode == "triangular":
        y[m] = 1 - np.abs(x[m])
    else:
        raise ValueError(f"{mode=} unknown.")

    return y


def _equivalent_kernel_grid(signal_mode: SignalMode,
                            kernel_mode: str = "quartic",
                            n_grid: int = 20_001) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate a fine-grid approximation of the equivalent kernel K^*.

    :param signal_mode: Either 'lvl' for d = mu - int mu or 'drv' for d = mu'.
    :param kernel_mode: Kernel type used for the base kernel.
    :param n_grid: Number of grid points used on [-1, 1].
    :return: Tuple containing grid points and equivalent-kernel values.
    :raises ValueError: If signal_mode is neither 'lvl' nor 'drv'.
    """
    x = np.linspace(-1, 1, n_grid)
    k = _kernel_eval(x, kernel_mode)

    if signal_mode == "lvl":
        k_star = 2 * np.sqrt(2) * _kernel_eval(np.sqrt(2) * x, kernel_mode) - k
    elif signal_mode == "drv":
        k_star = x * k
    else:
        raise ValueError("signal_mode must be either 'lvl' or 'drv'.")

    return x, k_star


def _equivalent_kernel_discrete(bw: int,
                                signal_mode: SignalMode,
                                kernel_mode: str = "quartic") -> np.ndarray:
    """
    Calculate the discrete equivalent kernel used in the Gaussian calibration.

    :param bw: Bandwidth of the local linear or Jackknife estimator as integer.
    :param signal_mode: Either 'lvl' for d = mu - int mu or 'drv' for d = mu'.
    :param kernel_mode: Kernel type used for the base kernel.
    :return: Discrete equivalent kernel as NumPy array.
    :raises ValueError: If signal_mode is neither 'lvl' nor 'drv'.
    """
    x = np.arange(-bw, bw + 1) / bw
    k = _kernel_eval(x, kernel_mode)

    if signal_mode == "lvl":
        return 2 * np.sqrt(2) * _kernel_eval(np.sqrt(2) * x, kernel_mode) - k
    if signal_mode == "drv":
        return x * k

    raise ValueError("signal_mode must be either 'lvl' or 'drv'.")


def _kernel_constants(signal_mode: SignalMode,
                      kernel_mode: str = "quartic") -> tuple[float, float]:
    """
    Calculate ||K^*||_2 and Lambda_2 for the extreme-value centering.

    :param signal_mode: Either 'lvl' for d = mu - int mu or 'drv' for d = mu'.
    :param kernel_mode: Kernel type used for the base kernel.
    :return: Tuple containing ||K^*||_2 and Lambda_2.
    """
    x, k_star = _equivalent_kernel_grid(signal_mode, kernel_mode)
    dk_star = np.gradient(k_star, x)

    norm = np.sqrt(np.trapz(k_star ** 2, x))
    lambda_2 = np.sqrt(np.trapz(dk_star ** 2, x) / np.trapz(k_star ** 2, x))

    return norm, lambda_2


def _window_size(n: int, delta: float) -> int:
    """
    Convert the minimal stable-state duration delta into a grid window length.

    :param n: Number of observations.
    :param delta: Minimal stable-state duration as fraction of [0, 1].
    :return: Window length as integer.
    :raises ValueError: If delta is not in [0, 1].
    """
    if not 0 <= delta <= 1:
        raise ValueError("delta must be in [0, 1].")

    return max(1, int(np.floor(delta * (n - 1))) + 1)


def _moving_window_max_abs(x: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate moving-window suprema of the absolute signal.

    :param x: Signal values on the observation grid.
    :param window: Window length as integer.
    :return: Moving-window maxima of |x|.
    """
    if window == 1:
        return np.abs(x)

    windows = np.lib.stride_tricks.sliding_window_view(np.abs(x), window)

    return np.max(windows, axis=-1)


def _stable_deviation(d: np.ndarray, delta: float) -> tuple[float, np.ndarray]:
    """
    Calculate the plug-in stable-state deviation d_inf.

    :param d: Estimated stability signal.
    :param delta: Minimal stable-state duration as fraction of [0, 1].
    :return: Tuple containing d_inf_hat and all moving-window deviations.
    """
    window = _window_size(len(d), delta)
    moving = _moving_window_max_abs(d, window)

    return float(np.min(moving)), moving


def _near_extremal_mask(d: np.ndarray,
                        d_inf: float,
                        moving: np.ndarray,
                        delta: float,
                        phi: float) -> np.ndarray:
    """
    Estimate the near-extremal set used for localized calibration.

    :param d: Estimated stability signal.
    :param d_inf: Estimated stable-state deviation.
    :param moving: Moving-window deviations corresponding to d.
    :param delta: Minimal stable-state duration as fraction of [0, 1].
    :param phi: Localization tolerance for the near-extremal set.
    :return: Boolean mask of the estimated near-extremal set.
    """
    n = len(d)
    window = _window_size(n, delta)

    e1 = np.abs(np.abs(d) - d_inf) <= phi
    e3 = moving - d_inf <= phi
    e2 = np.convolve(e3.astype(int), np.ones(window, dtype=int),
                     mode="full")[:n] > 0

    return e1 & e2


def _components(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Convert a Boolean mask into contiguous index components.

    :param mask: Boolean mask on a one-dimensional grid.
    :return: List of tuples containing start and end indices of components.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    cuts = np.flatnonzero(np.diff(idx) > 1) + 1
    parts = np.split(idx, cuts)

    return [(int(p[0]), int(p[-1])) for p in parts]


def _a_from_w(w: float, ell: float) -> float:
    """
    Calculate the corrected centering constant a_n from W_n and ell_n.

    :param w: Effective sample size W_n.
    :param ell: Preliminary logarithmic centering ell_n.
    :return: Corrected centering constant a_n.
    """
    log_w = max(float(np.log(max(w, 1.0))), 0.0)
    log_wc = ell * np.sqrt(ell ** 2 + 2 * log_w) - ell ** 2

    return ell + log_wc / ell


def _effective_sample_size(mask: np.ndarray,
                           sigma: np.ndarray,
                           h_n: float,
                           rho_n: float,
                           lambda_2: float) -> tuple[float, float, float, float]:
    """
    Calculate W_n, ell_n, a_n and the maximal local long-run standard deviation.

    :param mask: Boolean mask of the estimated near-extremal set.
    :param sigma: Estimated local long-run standard deviation.
    :param h_n: Bandwidth of the signal estimator on the [0, 1] scale.
    :param rho_n: Block length used in the effective sample size.
    :param lambda_2: Kernel curvature constant Lambda_2.
    :return: Tuple containing W_n, ell_n, a_n and sigma_max.
    :raises ValueError: If the near-extremal set is empty or rho_n is too small.
    """
    if not np.any(mask):
        raise ValueError("The estimated near-extremal set is empty.")

    arg = lambda_2 * rho_n / (2 * np.pi * h_n)
    if arg <= 1:
        raise ValueError(
            "lambda_2 * rho_n / (2*pi*h_n) must be larger than 1."
        )

    n = len(mask)
    ell = np.sqrt(2 * np.log(arg))

    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    sigma_max = float(np.max(sigma[mask]))

    step = max(1, int(round((rho_n + 2 * h_n) * n)))
    reps = []

    for a, b in _components(mask):
        comp = np.arange(a, b + 1)
        if comp.size <= step:
            reps.append(comp[comp.size // 2])
        else:
            reps.extend(comp[::step])

    reps = np.asarray(reps, dtype=int)
    weights = np.exp(
        -0.5 * ell ** 2 * ((sigma_max ** 2 / sigma[reps] ** 2) - 1)
    )

    w = float(np.sum(weights))
    a_n = _a_from_w(w, ell)

    return w, ell, a_n, sigma_max


def _estimate_signal(X: np.ndarray,
                     bw: int,
                     mode: SignalMode,
                     g: float | None = None) -> tuple:
    """
    Estimate the mean, stability signal and pointwise rate normalization.

    :param X: Time series given as NumPy array of shape (n_time,).
    :param bw: Bandwidth of the local linear or Jackknife estimator as integer.
    :param mode: Either 'lvl' for d = mu - int mu or 'drv' for d = mu'.
    :return: Tuple containing mu_hat, d_hat and r_n.
    :raises ValueError: If mode is neither 'lvl' nor 'drv'.
    """
    n = len(X)
    h_n = bw / n

    if mode == "lvl":
        mu = jackknife_estimator(X, bw)[0]
        d = mu - np.mean(mu) if g is None else mu - g
        r_n = np.sqrt(n * h_n)
    elif mode == "drv":
        mu, d = local_linear_estimator(X, bw)
        r_n = np.sqrt(n * h_n ** 3)
    else:
        raise ValueError("mode must be either 'lvl' or 'drv'.")

    return mu, d, r_n


def _gaussian_calibration(d: np.ndarray,
                          mask: np.ndarray,
                          sigma: np.ndarray,
                          bw: int,
                          h_n: float,
                          a_n: float,
                          sigma_max: float,
                          norm_k: float,
                          mode: SignalMode,
                          alpha: float,
                          kernel_mode: str,
                          n_boot: int,
                          seed: int | None) -> tuple[float, np.ndarray]:
    """
    Calculate Gaussian-calibrated critical value and simulated test statistics. [main.pdf]

    :param d: Estimated stability signal.
    :param mask: Boolean mask of the estimated near-extremal set.
    :param sigma: Estimated local long-run standard deviation.
    :param bw: Bandwidth of the signal estimator as integer.
    :param h_n: Bandwidth of the signal estimator on the [0, 1] scale.
    :param a_n: Corrected centering constant.
    :param sigma_max: Maximal sigma value on the near-extremal set.
    :param norm_k: L2 norm of the equivalent kernel K*.
    :param mode: Either 'lvl' for d = mu - int mu or 'drv' for d = mu'.
    :param alpha: Significance level.
    :param kernel_mode: Kernel type used for the base kernel.
    :param n_boot: Number of Gaussian multiplier draws.
    :param seed: Random seed used for reproducibility.
    :return: Tuple containing the critical value and simulated statistics.
    """
    rng = np.random.default_rng(seed)
    n = len(d)
    k_star = _equivalent_kernel_discrete(bw, mode, kernel_mode)

    signs = np.sign(d)
    signs[signs == 0] = 1

    sim_stats = np.empty(n_boot)

    for b in range(n_boot):
        y = sigma * rng.standard_normal(n)
        g = signal.convolve(y, k_star, mode="same") / np.sqrt(n * h_n)
        z = np.max(-signs[mask] * g[mask])
        sim_stats[b] = a_n * z / (norm_k * sigma_max) - a_n ** 2

    critical_value = float(np.quantile(sim_stats, 1 - alpha))

    return critical_value, sim_stats


def _tune_signal_bandwidth(X: np.ndarray,
                           mode: SignalMode,
                           min_bw: int | None = None,
                           max_bw: int | None = None,
                           num_folds: int = 5,
                           step_size: int = 1) -> int:
    """
    Tune the bandwidth of the mean estimator by cross validation.

    :param X: Time series given as NumPy array of shape (n_time,).
    :param mode: Either 'lvl' for d = mu - int mu or 'drv' for d = mu'.
    :param min_bw: Minimal candidate bandwidth. If None, a default is used.
    :param max_bw: Maximal candidate bandwidth. If None, a default is used.
    :param num_folds: Number of folds used for cross validation.
    :param step_size: Step size between candidate bandwidths.
    :return: Cross-validation selected bandwidth as integer.
    :raises ValueError: If mode is neither 'lvl' nor 'drv'.
    """
    n = len(X)

    if min_bw is None:
        min_bw = max(3, int(np.floor(0.03 * n)))
    if max_bw is None:
        max_bw = max(min_bw, int(np.floor(0.25 * n)))

    if mode == "lvl":
        def estimator(y, bw, filter_array):
            return jackknife_estimator(y, bw, filter_array=filter_array)[0]
    elif mode == "drv":
        def estimator(y, bw, filter_array):
            return local_linear_estimator(y, bw, filter_array=filter_array)[0]
    else:
        raise ValueError("mode must be either 'lvl' or 'drv'.")

    return int(np.squeeze(bandwidth_cv(
        X, min_bw=min_bw, max_bw=max_bw, estimator=estimator,
        num_folds=num_folds, step_size=step_size
    )))


def _default_rho_phi(d: np.ndarray,
                     h_n: float,
                     r_n: float,
                     lambda_2: float,
                     rho: float | None = None,
                     phi: float | None = None) -> tuple[float, float]:
    """
    Set default values for rho and phi if they are not provided.

    :param d: Estimated stability signal.
    :param h_n: Bandwidth of the signal estimator on the [0, 1] scale.
    :param r_n: Pointwise rate normalization of the signal estimator.
    :param lambda_2: Kernel curvature constant Lambda_2.
    :param rho: Optional user-provided block length.
    :param phi: Optional user-provided localization tolerance.
    :return: Tuple containing rho and phi.
    """
    if rho is None:
        rho = max(np.sqrt(h_n), 4 * 2 * np.pi * h_n / lambda_2)
        rho = min(rho, 0.5)

    if lambda_2 * rho / (2 * np.pi * h_n) <= 1:
        rho = 1.05 * 2 * np.pi * h_n / lambda_2

    if phi is None:
        sd = np.nanstd(d)
        scale = sd if sd > 0 else max(np.nanstd(np.abs(d)), 1.0)
        stochastic = 2 * scale * np.sqrt(abs(np.log(h_n))) / max(r_n, 1e-12)
        deterministic = 0.05 * max(np.nanmax(d) - np.nanmin(d), scale, 1e-12)
        phi = max(stochastic, deterministic, 1e-10)

    return float(rho), float(phi)


def _first_stable_onsets(moving: np.ndarray,
                         deltas: np.ndarray,
                         n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate first tolerance-stable onsets for multiple tolerance values. [main.pdf]

    :param moving: Moving-window deviations of the estimated stability signal.
    :param deltas: Stability tolerances Delta.
    :param n: Number of observations.
    :return: Tuple containing onset indices and onset locations in [0, 1].
    """
    onset_idx = np.full(len(deltas), -1, dtype=int)
    onset_loc = np.full(len(deltas), np.nan, dtype=float)

    for j, Delta in enumerate(deltas):
        starts = np.flatnonzero(moving <= Delta)
        if starts.size:
            onset_idx[j] = int(starts[0])
            onset_loc[j] = onset_idx[j] / (n - 1)

    return onset_idx, onset_loc


def stable_state_test(X: np.ndarray,
                      Delta: float | np.ndarray,
                      delta: float,
                      bw: int | None = None,
                      g: float | None = None,
                      rho: float | None = None,
                      phi: float | None = None,
                      mode: SignalMode = "lvl",
                      alpha: float = 0.05,
                      calibration: Calibration = "both",
                      kernel_mode: str = "quartic",
                      tau_n: float | None = None,
                      tune_bw: bool = False,
                      min_bw: int | None = None,
                      max_bw: int | None = None,
                      num_folds: int = 5,
                      step_size: int = 1,
                      n_boot: int = 1_000,
                      seed: int | None = None) -> dict:
    """
    Perform stable-state tests H0: d_inf >= Delta against H1: d_inf < Delta. [main.pdf]

    :param X: Time series given as NumPy array of shape (n_time,).
    :param Delta: One or multiple stability tolerances used in the null
        hypotheses.
    :param delta: Minimal stable-state duration as fraction of [0, 1].
    :param bw: Bandwidth of the local linear or Jackknife estimator as integer.
        If None or tune_bw=True, the bandwidth is selected by cross validation.
    :param g: Constant to subtract from signal, if d = \mu.
    :param rho: Block length rho_n used in the effective sample size. If None,
        a finite-sample default is used.
    :param phi: Localization tolerance used to estimate the near-extremal set.
        If None, a finite-sample default is used.
    :param mode: Either 'lvl' for d = mu - int mu or 'drv' for d = mu'.
    :param alpha: Significance level.
    :param calibration: Either 'gumbel' or 'gaussian'.
    :param kernel_mode: Kernel type used for the base kernel.
    :param tau_n: Bandwidth of the local long-run variance estimator.
    :param tune_bw: Indicates whether bw is selected by cross validation.
    :param min_bw: Minimal candidate bandwidth for cross validation.
    :param max_bw: Maximal candidate bandwidth for cross validation.
    :param num_folds: Number of folds used for bandwidth cross validation.
    :param step_size: Step size between candidate bandwidths.
    :param n_boot: Number of Gaussian multiplier draws if calibration='gaussian'.
    :param seed: Random seed used for Gaussian calibration.
    :return: Dictionary containing test decisions, statistics, critical value,
        bandwidth, onset estimates and fitted objects.
    :raises ValueError: If calibration is neither 'gumbel' nor 'gaussian'.
    """
    scalar_input = np.ndim(Delta) == 0
    X = np.asarray(X, dtype=float)
    n = len(X)
    deltas = np.atleast_1d(np.asarray(Delta, dtype=float))

    if tune_bw or bw is None:
        bw = _tune_signal_bandwidth(
            X, mode=mode, min_bw=min_bw, max_bw=max_bw,
            num_folds=num_folds, step_size=step_size
        )

    h_n = bw / n
    mu, d, r_n = _estimate_signal(X, bw, mode, g)

    norm_k, lambda_2 = _kernel_constants(mode, kernel_mode)
    rho, phi = _default_rho_phi(d, h_n, r_n, lambda_2, rho=rho, phi=phi)

    lrv = estimate_long_run_variance(X, tau_n=tau_n, mu=mu)
    sigma = np.sqrt(np.maximum(lrv, 1e-12))

    d_inf, moving = _stable_deviation(d, delta)
    mask = _near_extremal_mask(d, d_inf, moving, delta, phi)

    w_n, ell_n, a_n, sigma_max = _effective_sample_size(
        mask, sigma, h_n, rho, lambda_2
    )

    statistics = a_n * r_n * (deltas - d_inf) / (norm_k * sigma_max) - a_n ** 2

    if calibration == "gumbel":
        critical_value = -np.log(-np.log(1 - alpha))
        reject = statistics > critical_value
        reject = bool(reject[0]) if scalar_input else reject
        p_values = 1 - np.exp(-np.exp(-statistics))
    elif calibration == "gaussian":
        critical_value, sim_stats = _gaussian_calibration(
            d=d,
            mask=mask,
            sigma=sigma,
            bw=bw,
            h_n=h_n,
            a_n=a_n,
            sigma_max=sigma_max,
            norm_k=norm_k,
            mode=mode,
            alpha=alpha,
            kernel_mode=kernel_mode,
            n_boot=n_boot,
            seed=seed,
        )
        reject = statistics > critical_value
        reject = bool(reject[0]) if scalar_input else reject
        p_values = np.array([
            (1 + np.sum(sim_stats >= stat)) / (n_boot + 1)
            for stat in statistics
        ])
    elif calibration == "both":
        critical_value_gumbel = -np.log(-np.log(1 - alpha))
        critical_value_gaussian, sim_stats = _gaussian_calibration(
            d=d,
            mask=mask,
            sigma=sigma,
            bw=bw,
            h_n=h_n,
            a_n=a_n,
            sigma_max=sigma_max,
            norm_k=norm_k,
            mode=mode,
            alpha=alpha,
            kernel_mode=kernel_mode,
            n_boot=n_boot,
            seed=seed,
        )
        reject_gumbel = statistics > critical_value_gumbel
        reject_gaussian = statistics > critical_value_gaussian

        p_values_gumbel = 1 - np.exp(-np.exp(-statistics))
        p_values_gaussian = np.array([
            (1 + np.sum(sim_stats >= stat)) / (n_boot + 1)
            for stat in statistics
        ])

        reject = {'gumbel': reject_gumbel, 'gaussian': reject_gaussian}
        p_values = {'gumbel': p_values_gumbel, 'gaussian': p_values_gaussian}
        critical_value = {'gumbel': critical_value_gumbel,
                          'gaussian': critical_value_gaussian}
    else:
        raise ValueError("calibration must be either 'gumbel' or 'gaussian'.")

    onset_idx, onset_loc = _first_stable_onsets(moving, deltas, n)
    opt_onset_idx, opt_onset_loc = estimate_stable_onset(d, delta)

    return {
        "reject": reject,
        "p_value": p_values,
        "statistic": float(statistics[0]) if scalar_input else statistics,
        "critical_value": critical_value,
        "d_inf_hat": float(d_inf),
        "Delta": float(deltas[0]) if scalar_input else deltas,
        "delta": float(delta),
        "mode": mode,
        "calibration": calibration,
        "alpha": float(alpha),
        "bw": int(bw),
        "h_n": float(h_n),
        "rho": float(rho),
        "phi": float(phi),
        "a_n": float(a_n),
        "ell_n": float(ell_n),
        "W_n": float(w_n),
        "sigma_max": float(sigma_max),
        "onset_index": int(onset_idx[0]) if scalar_input else onset_idx,
        "onset_location": float(onset_loc[0]) if scalar_input else onset_loc,
        "optimal_onset_index": int(opt_onset_idx),
        "optimal_onset_location": float(opt_onset_loc),
        "extremal_mask": mask,
        "d_hat": d,
        "mu_hat": mu,
        "sigma_hat": sigma,
        "moving_deviation": moving,
    }


def estimate_stable_onset(d: np.ndarray,
                          delta: float,
                          threshold: float | None = None) -> tuple[int, float]:
    """
    Estimate the onset of a stable or near-optimal stable interval.

    :param d: Estimated stability signal.
    :param delta: Minimal stable-state duration as fraction of [0, 1].
    :param threshold: Optional tolerance threshold Delta. If None, the first
        minimax-optimal interval is returned.
    :return: Tuple containing the onset index and the onset location in [0, 1].
    """
    d_inf, moving = _stable_deviation(d, delta)

    if threshold is None:
        starts = np.flatnonzero(moving == np.min(moving))
    else:
        starts = np.flatnonzero(moving <= threshold)

    if starts.size == 0:
        return -1, np.nan

    j = int(starts[0])

    return j, j / (len(d) - 1)


def calculate_test_statistic(X: np.ndarray,
                             h: float | None = None,
                             rho: float | None = None,
                             phi: float | None = None,
                             bw: int | None = None,
                             g: float | None = None,
                             mode: SignalMode = "lvl",
                             *,
                             Delta: float | np.ndarray,
                             delta: float = 0.0,
                             kernel_mode: str = "quartic",
                             tau_n: float | None = None,
                             tune_bw: bool = False,
                             min_bw: int | None = None,
                             max_bw: int | None = None,
                             num_folds: int = 5,
                             step_size: int = 1) -> float | np.ndarray:
    """
    Calculate normalized stable-state test statistics for one or multiple Delta. [main.pdf]

    :param X: Time series given as NumPy array of shape (n_time,).
    :param h: Bandwidth on the [0, 1] scale. If None, bw / n is used.
    :param rho: Block length rho_n used in the effective sample size. If None,
        a finite-sample default is used.
    :param phi: Localization tolerance used to estimate the near-extremal set.
        If None, a finite-sample default is used.
    :param bw: Bandwidth of the local linear or Jackknife estimator as integer.
        If None or tune_bw=True, the bandwidth is selected by cross validation.
    :param g: Constant to subtract from signal, if d = \mu.
    :param mode: Either 'lvl' for d = mu - int mu or 'drv' for d = mu'.
    :param Delta: One or multiple stability tolerances.
    :param delta: Minimal stable-state duration as fraction of [0, 1].
    :param kernel_mode: Kernel type used for the base kernel.
    :param tau_n: Bandwidth of the local long-run variance estimator.
    :param tune_bw: Indicates whether bw is selected by cross validation.
    :param min_bw: Minimal candidate bandwidth for cross validation.
    :param max_bw: Maximal candidate bandwidth for cross validation.
    :param num_folds: Number of folds used for bandwidth cross validation.
    :param step_size: Step size between candidate bandwidths.
    :return: Stable-state test statistic or statistics.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    deltas = np.atleast_1d(np.asarray(Delta, dtype=float))

    if tune_bw or bw is None:
        bw = _tune_signal_bandwidth(
            X, mode=mode, min_bw=min_bw, max_bw=max_bw,
            num_folds=num_folds, step_size=step_size
        )

    h_n = bw / n if h is None else h
    mu, d, r_n = _estimate_signal(X, bw, mode, g)

    norm_k, lambda_2 = _kernel_constants(mode, kernel_mode)
    rho, phi = _default_rho_phi(d, h_n, r_n, lambda_2, rho=rho, phi=phi)

    lrv = estimate_long_run_variance(X, tau_n=tau_n, mu=mu)
    sigma = np.sqrt(np.maximum(lrv, 1e-12))

    d_inf, moving = _stable_deviation(d, delta)
    mask = _near_extremal_mask(d, d_inf, moving, delta, phi)

    _, _, a_n, sigma_max = _effective_sample_size(
        mask, sigma, h_n, rho, lambda_2
    )

    statistics = a_n * r_n * (deltas - d_inf) / (norm_k * sigma_max) - a_n ** 2

    return float(statistics[0]) if np.ndim(Delta) == 0 else statistics


def _sanity_check_stable_state_test(seed: int = 123) -> None:
    """
    Run shape and functionality checks for the stable-state implementation.

    :param seed: Random seed used for reproducibility.
    :return: None.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Exact check of the minimax stable-state functional and onset.
    # ------------------------------------------------------------------
    d = np.array([2.0, 2.0, 0.1, 0.2, 0.1, 2.0])
    delta = 0.4
    window = _window_size(len(d), delta)

    assert window == 3

    d_inf, moving = _stable_deviation(d, delta)

    np.testing.assert_allclose(moving, np.array([2.0, 2.0, 0.2, 2.0]))
    np.testing.assert_allclose(d_inf, 0.2)

    onset_idx, onset_loc = estimate_stable_onset(d, delta)
    assert onset_idx == 2
    np.testing.assert_allclose(onset_loc, 2 / (len(d) - 1))

    onset_idx_thr, onset_loc_thr = estimate_stable_onset(
        d, delta, threshold=0.25
    )
    assert onset_idx_thr == 2
    np.testing.assert_allclose(onset_loc_thr, 2 / (len(d) - 1))

    onset_idx_none, onset_loc_none = estimate_stable_onset(
        d, delta, threshold=0.15
    )
    assert onset_idx_none == -1
    assert np.isnan(onset_loc_none)

    mask = _near_extremal_mask(d, d_inf, moving, delta, phi=1e-12)
    np.testing.assert_array_equal(
        mask, np.array([False, False, False, True, False, False])
    )

    # ------------------------------------------------------------------
    # 2. Exact check of first threshold-stable onsets for multiple Delta.
    # ------------------------------------------------------------------
    Delta_grid = np.array([0.15, 0.25, 2.0])
    onset_idx, onset_loc = _first_stable_onsets(moving, Delta_grid, len(d))

    np.testing.assert_array_equal(onset_idx, np.array([-1, 2, 0]))
    assert np.isnan(onset_loc[0])
    np.testing.assert_allclose(onset_loc[1:], np.array([2 / 5, 0.0]))

    # ------------------------------------------------------------------
    # 3. Test statistic must be strictly increasing in Delta.
    # ------------------------------------------------------------------
    n = 450
    t = np.linspace(0, 1, n)

    X_const = 1.0 + 0.04 * rng.standard_normal(n)
    Delta_grid = np.array([0.0, 0.10, 0.25, 0.50])

    res_const = stable_state_test(
        X_const, Delta=Delta_grid, delta=0.15, bw=45, mode="lvl",
        calibration="gumbel", rho=0.25, phi=0.04
    )

    statistics = np.asarray(res_const["statistic"])
    rejects = np.asarray(res_const["reject"], dtype=bool)

    assert statistics.shape == Delta_grid.shape
    assert rejects.shape == Delta_grid.shape
    assert np.all(np.diff(statistics) > 0)
    assert not rejects[0]
    assert rejects[-1]
    assert res_const["d_inf_hat"] < 0.15
    assert res_const["optimal_onset_index"] >= 0
    assert res_const["onset_index"][0] == -1
    assert res_const["onset_index"][-1] == 0

    # ------------------------------------------------------------------
    # 4. Level-stability functionality: constant signal is stable early.
    # ------------------------------------------------------------------
    res_lvl = stable_state_test(
        X_const, Delta=0.20, delta=0.20, bw=45, mode="lvl",
        calibration="gumbel", rho=0.25, phi=0.04
    )

    assert res_lvl["reject"]
    assert res_lvl["d_inf_hat"] < 0.20
    assert 0 <= res_lvl["onset_index"] < int(0.15 * n)
    assert 0 <= res_lvl["optimal_onset_index"] < int(0.25 * n)

    # ------------------------------------------------------------------
    # 5. Derivative-stability functionality: plateau onset is detected late.
    # ------------------------------------------------------------------
    mu_plateau = np.where(t < 0.45, 2.5 * t, 2.5 * 0.45)
    X_plateau = mu_plateau + 0.02 * rng.standard_normal(n)

    res_drv = stable_state_test(
        X_plateau, Delta=np.array([0.05, 0.15, 0.40]), delta=0.15,
        bw=55, mode="drv", calibration="gumbel", rho=0.25, phi=0.20
    )

    drv_stats = np.asarray(res_drv["statistic"])
    drv_rejects = np.asarray(res_drv["reject"], dtype=bool)
    drv_onsets = np.asarray(res_drv["onset_location"])

    assert np.all(np.diff(drv_stats) > 0)
    assert np.all(np.diff(drv_rejects.astype(int)) >= 0)
    assert res_drv["optimal_onset_location"] > 0.30
    assert res_drv["optimal_onset_location"] < 0.80
    assert drv_rejects[-1]
    assert np.isnan(drv_onsets[0]) or drv_onsets[0] > 0.25
    assert drv_onsets[-1] > 0.25

    # ------------------------------------------------------------------
    # 6. Onset monotonicity: larger Delta cannot move first onset later.
    # ------------------------------------------------------------------
    valid = drv_onsets[~np.isnan(drv_onsets)]
    assert np.all(np.diff(valid) <= 1e-12)

    # ------------------------------------------------------------------
    # 7. calculate_test_statistic must agree with stable_state_test.
    # ------------------------------------------------------------------
    stat = calculate_test_statistic(
        X_const, Delta=Delta_grid, delta=0.15, bw=45, mode="lvl",
        rho=0.25, phi=0.04
    )

    np.testing.assert_allclose(stat, res_const["statistic"])

    # ------------------------------------------------------------------
    # 8. Cross-validation path must choose an admissible bandwidth and run.
    # ------------------------------------------------------------------
    res_cv = stable_state_test(
        X_const, Delta=0.20, delta=0.15, bw=None, tune_bw=True,
        min_bw=20, max_bw=40, step_size=10, num_folds=3,
        mode="lvl", calibration="gumbel", rho=0.25, phi=0.04
    )

    assert 20 <= res_cv["bw"] <= 40
    assert res_cv["bw"] in {20, 30, 40}
    assert np.isfinite(res_cv["statistic"])
    assert isinstance(res_cv["reject"], bool)

    # ------------------------------------------------------------------
    # 9. Gaussian calibration must produce a finite critical value.
    # ------------------------------------------------------------------
    res_gauss = stable_state_test(
        X_const, Delta=0.20, delta=0.15, bw=45, mode="lvl",
        calibration="gaussian", rho=0.25, phi=0.04,
        n_boot=100, seed=seed
    )

    assert np.isfinite(res_gauss["critical_value"])
    assert np.isfinite(res_gauss["statistic"])
    assert res_gauss["critical_value"] != res_lvl["critical_value"]

    print("All functional sanity checks passed.")


if __name__ == '__main__':
    _sanity_check_stable_state_test()