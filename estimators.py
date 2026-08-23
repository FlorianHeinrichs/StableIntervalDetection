#
# tests/estimators.py
#
# Project: Detecting Stable States in Non-Stationary Time Series
# Date: 2026-07-27
# Author: Florian Heinrichs
#
# Long run variance estimator from:
# "Bücher, A., Dette, H., & Heinrichs, F. (2021). Are deviations in a
# gradually varying mean relevant? A testing approach based on sup-norm
# estimators. The Annals of Statistics, 49(6), 3583-3617."


from typing import Callable

import numpy as np
from scipy import signal


def estimate_long_run_variance(X: np.ndarray, tau_n: float = None,
                               mu: np.ndarray = None, h: int = 10,
                               h_n: float = None) -> np.ndarray:
    """
    Function to estimate the (stationary) long-run variance of a given time
    series. If multiple time series are provided, the calculations are done in
    parallel.

    :param X: Time series given as numpy array.
    :param tau_n: Bandwidth of estimator of \sigma.
    :param mu: Sequence of means (possibly non-stationary).
    :param h: Number of auto-covariances to use for tuning bandwidth of
        estimator.
    :param h_n: Bandwidth of estimator of \mu.
    :return: Long-run variance(s) given as numpy array.
    """
    n = len(X)

    tau_n = n ** (- 1 / 4) if tau_n is None else tau_n
    h_n = n ** (- 1 / 7) if h_n is None else h_n
    mu = jackknife_estimator(X, int(n * h_n))[0] if mu is None else mu
    X_c = X - mu

    # Calculate first h (co-)variances to tune bandwidth m
    variance = np.var(X_c)
    covariances = [variance]

    for k in range(1, h + 1):
        X_start = X_c[:-k] - np.mean(X_c[:-k], axis=-1, keepdims=True)
        X_end = X_c[k:] - np.mean(X_c[k:], axis=-1, keepdims=True)
        cov = np.sum(X_start * X_end, axis=-1) / (n - h)
        covariances.append(cov)

    covariances = np.array(covariances)

    # Calculate m (kernel size for partial sums)
    m_n = np.maximum(
        np.floor(np.sqrt(1 - variance / np.sum(np.abs(covariances)))
                 * n ** (1 / 3)).astype(int), 1
    )

    # Calculate long-run variance estimator
    bw = int(tau_n * n)
    kernel_support, kernel = get_kernel(bw)

    # Create padded cumulative sums
    X_padded = np.pad(X, (m_n, m_n), mode='edge')
    S = np.cumsum(X_padded)

    i = np.arange(n)
    left = S[i + m_n] - S[i]
    right = S[i + 2 * m_n] - S[i + m_n]
    diff = left - right
    numerator = (diff ** 2) / (2 * m_n)

    S0 = convolve(numerator, kernel)
    S1 = convolve(np.ones_like(numerator), kernel)

    lrv_estimator = S0 / S1
    lrv_estimator = np.pad(lrv_estimator, (bw, bw), mode='edge')

    return lrv_estimator


def jackknife_estimator(X: np.ndarray,
                        bw: int,
                        filter_array: np.ndarray = None) -> tuple:
    """
    Function to calculate the Jackknife version of the local linear estimators.

    :param X: Time series given as numpy array.
    :param bw: Bandwidth of the estimator.
    :param filter_array: Filter array to leave out certain observations (used
        for cross validation).
    :return: Returns the Jackknife estimators given as numpy array.
    """
    mu_hat, mu_prime_hat = local_linear_estimator(
        X, bw, filter_array=filter_array
    )

    mu_hat2, mu_prime_hat2 = local_linear_estimator(
        X, int(bw / np.sqrt(2)), filter_array=filter_array
    )

    mu_tilde = 2 * mu_hat2 - mu_hat
    mu_prime_tilde = (np.sqrt(2) / (np.sqrt(2) - 1) * mu_prime_hat2
                      - mu_prime_hat / (np.sqrt(2) - 1))

    return mu_tilde, mu_prime_tilde


def local_linear_estimator(X: np.ndarray,
                           bw: int,
                           filter_array: np.ndarray = None) -> tuple:
    """
    Use local linear regression to estimate mu and its Frechet derivative.

    :param X: Time series given as numpy array.
    :param bw: Bandwidth of the estimator as int.
    :param filter_array: Filter array to leave out certain observations (used
        for cross validation).
    :return: Returns the local linear estimators.
    """
    spatial_dims = len(X.shape) - 1
    n_time = X.shape[0]

    if filter_array is None:
        filter_array = np.ones_like(X, dtype=bool)

    X_filtered = X.copy()
    X_filtered[~filter_array] = 0

    X_support = np.ones_like(X_filtered)
    X_support[~filter_array] = 0

    kernel_support, kernel = get_kernel(bw)
    kernel = kernel.reshape((1,) * spatial_dims + (-1,))
    supp_kern = kernel_support * kernel
    supp2_kern = kernel_support ** 2 * kernel

    padding = ((bw, bw),) + spatial_dims * ((0, 0),)
    X_filtered = np.pad(X_filtered, padding, mode='edge')
    X_support = np.pad(X_support, padding, mode='edge')

    S0 = convolve(X_support, kernel)
    S1 = convolve(X_support, supp_kern[..., ::-1])
    S2 = convolve(X_support, supp2_kern[..., ::-1])

    R0 = convolve(X_filtered, kernel)
    R1 = convolve(X_filtered, supp_kern[..., ::-1])

    denominator = S0 * S2 - S1 ** 2
    mu_hat = (S2 * R0 - S1 * R1) / (denominator + 1e-10)
    mu_prime_hat = (S0 * R1 - S1 * R0) / (bw / n_time * denominator + 1e-10)

    return mu_hat, mu_prime_hat


def get_kernel(bw: int,
               mode: str = 'quartic',
               version: str = 'regular') -> (np.ndarray, np.ndarray):
    """
    Define kernel for kernel based estimators.

    :param bw: Bandwidth of the estimator as int.
    :param mode: Mode of the kernel given as string. Currently only the
        'quartic', 'triweight', and 'tricube' kernels are supported.
    :param version: Version of the kernel, either 'regular' or 'jackknife'.
    :return: Returns the kernel and its support as numpy arrays.
    :raises: ValueError if unsupported mode is chosen.
    """
    if version == 'regular':
        support = np.arange(-bw, bw + 1) / bw

        if mode == 'quartic':
            kernel = 15 / 16 * (1 - support ** 2) ** 2
        elif mode == 'triweight':
            kernel = 35 / 32 * (1 - support ** 2) ** 3
        elif mode == 'tricube':
            kernel = 70 / 81 * (1 - np.abs(support) ** 3) ** 3
        elif mode == 'triangular':
            kernel = (1 - np.abs(support))
        else:
            raise ValueError(f"{mode=} unknown.")

    elif version == 'jackknife':
        bw2 = int(bw // np.sqrt(2))
        support, kern = get_kernel(bw)
        kern2 = np.sqrt(8) * get_kernel(bw2)[1]
        n_diff = (len(kern) - len(kern2)) // 2
        kernel = np.pad(kern2, (n_diff, n_diff)) - kern

    else:
        raise ValueError(f"{version=} unknown.")

    return support, kernel


def bandwidth_cv(X: np.ndarray,
                 min_bw: int,
                 max_bw: int,
                 estimator: Callable,
                 num_folds: int = 5,
                 step_size: int = 1,
                 batch_axis: int = -1,
                 return_mses: bool = False) -> np.ndarray | tuple:
    """
    Function to tune the bandwidth of kernel estimators.

    :param X: Functional time series given as numpy array.
    :param min_bw: Minimal bandwidth of the estimator.
    :param max_bw: Maximal bandwidth of the estimator.
    :param estimator: Kernel estimator whose bandwidth to tune.
    :param num_folds: Number of folds used for cross validation. Defaults to 5.
    :param step_size: Step size of bandwidth. Defaults to 1.
    :param batch_axis: The first axis of the NumPy array is the time axis along
        which the time series is smoothened, and the bandwidth is selected. The
        shape of the time series is (n_time,) + space_shape, and a single
        bandwidth, that is optimal across all points in space, is returned. If
        batch_axis is provided, the bandwidth is tuned for each entry along this
        axis separately.
    :param return_mses: Indicates whether MSEs are returned too.
    :return: Returns the optimal bandwidth as int.
    """
    indices_shuffle = np.arange(X.shape[0] // num_folds * num_folds)
    np.random.shuffle(indices_shuffle)
    folds = np.split(indices_shuffle, num_folds)
    indices = np.arange(X.shape[0])

    non_batch_axes = tuple(a for a in range(X.ndim) if a != batch_axis)
    n_samples = 1 if batch_axis == -1 else X.shape[batch_axis]

    best_bw, best_mse = - np.ones(n_samples, dtype=int), - np.ones(n_samples)
    mses = []

    for bw in range(min_bw, max_bw + 1, step_size):
        mse = np.zeros(n_samples)
        for fold in folds:
            filter_array = ~np.isin(indices, fold)
            estimate = estimator(X, bw, filter_array)

            mse += np.nanmean(
                (X[~filter_array] - estimate[~filter_array]) ** 2,
                axis=non_batch_axes
            )

        mses.append(mse)

        better_bw = np.where((mse < best_mse) | (best_mse == -1)
                             | np.isnan(best_mse))
        best_bw[better_bw], best_mse[better_bw] = bw, mse[better_bw]

    return (best_bw, mses) if return_mses else best_bw


def convolve(X: np.ndarray, kernel: np.ndarray, mode: str = 'valid') -> np.ndarray:
    """
    Convolve X with kernel across time axis.

    :param X: NumPy array of shape (n_time,) + space_shape.
    :param kernel: NumPy array of shape (bw,) with bw < n_time.
    :param mode: Mode of convolution, defaults to 'valid'.
    :return: Convolution of X with kernel across time axis.
    """
    convolution = np.moveaxis(
        signal.convolve(np.moveaxis(X, 0, -1), kernel, mode=mode), -1, 0
    )
    return convolution
