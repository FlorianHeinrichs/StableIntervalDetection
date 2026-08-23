#
# experiments.py
#
# Project: Detecting Stable States in Non-Stationary Time Series
# Date: 2026-07-30
# Author: Florian Heinrichs
#
# Script containing main experiments for test comparison.

from datetime import datetime
import json

import numpy as np

from data_generation import generate_data
from tests import stable_state_test


def experiment(n: int, n_samples: int, mean: str, delta: float,
               Delta: np.ndarray, error: str, sigma: str,
               alpha: float = 0.05) -> dict:
    """
    Main function for running the experiments.

    :param n: Length of generated time series.
    :param n_samples: Number of generated time series.
    :param mean: Type of mean in ['1', '2', 'abrupt', 'const'].
    :parma delta: Minimal length of stable interval.
    :param Delta: Scientifically relevant threshold(s).
    :param error: Type of error in ['iid', 'ar', 'ma'].
    :param sigma: Type of sigma in ['0', '1', '2', '3'].
    :param alpha: Level of test.
    :return: Dictionary with test results.
    """
    # Generate Data
    data = generate_data(n, n_samples, mean, delta, error, sigma)

    # Do tests
    onsets, optimal_onsets, rejects = [], [], []

    for x in data.transpose():
        result = stable_state_test(x, Delta, delta, mode=mean[:3], alpha=alpha,
                                   calibration="both", tune_bw=True)
        onsets.append(result['onset_location'])
        optimal_onsets.append(result['optimal_onset_location'])
        rejects.append(result['reject'])

    results = {'onsets': onsets, 'optimal_onsets': optimal_onsets,
               'rejects': rejects}
    return results


def get_config(n_samples) -> list:
    """
    Get list of configurations for experiments.

    :param n_samples: Number of generated time series.
    :return: List of configurations, where each entry of the list is a tuple
        of arguments of experiment().
    """
    deltas = [0, 0.1, 0.2, 1]
    Delta0 = 0.05 * np.arange(1, 11)
    Delta1 = 0.5 + 0.1 * np.arange(1, 11)
    mean_delta = {'drv_settling': Delta1,
                  'drv_settling_drift': Delta1,
                  'drv_plateau': Delta0,
                  'drv_tmp_plateau': Delta0,
                  'lvl_linear': Delta1,
                  'lvl_bump': Delta0}

    config = [
        (n, n_samples, mean, delta, Delta, error, sigma)
        for n in [100, 200, 500, 1000]
        for mean, Delta in mean_delta.items()
        for delta in deltas
        for error in ['iid', 'ar', 'ma', 'loc_stat']
        for sigma in ['0', '1', '2', '3']
    ]

    return config


def convert_to_serializable(data):
    if isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(x) for x in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, tuple):
        return [convert_to_serializable(value) for value in data]
    else:
        return data


if __name__ == '__main__':
    alpha = 0.05
    n_samples = 1000
    config = get_config(n_samples)
    results = {}

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = f"../results/samples{n_samples}_" + now + ".json"

    for args in config:
        arg_str = f"n{args[0]}_{args[2]}_delta{args[3]}_{args[5]}_sigma{args[6]}"
        print(f"Starting experiment: {arg_str}")
        results[arg_str] = experiment(*args, alpha=alpha)

        with open(filepath, 'w') as file:
            result_tmp = convert_to_serializable(results)
            json.dump(result_tmp, file, indent=4)
