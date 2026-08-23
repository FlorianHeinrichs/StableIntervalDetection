#
# data_application_vitaldb.py
#
# Project: Detecting Stable States in Non-Stationary Time Series
# Date: 2026-08-18
# Author: Florian Heinrichs
#
# Script containing experiments for VitalDB data.

import os

from vitaldb import VitalFile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import stable_state_test, convert_to_serializable


def isolate_mbp(folder_in: str, folder_out: str, freq: float = 0.1):
    for name in os.listdir(folder_in):
        path = os.path.join(folder_in, name)
        path_out = os.path.join(folder_out, name.replace('vital', 'csv'))

        if (os.path.isfile(path) and name.lower().endswith(".vital")
              and not os.path.exists(path_out)):
            vf = VitalFile(path, ['Solar8000/ART_MBP', 'Solar8000/NIBP_MBP'])
            tracks = list(vf.trks.keys())
            if len(tracks) > 0:
                x = vf.to_numpy(tracks, interval=1/freq)
                df = pd.DataFrame(x, columns=list(tracks))

                df.to_csv(path_out, index=False)


def display_vital(folder, start, end, column: str = None, i_range: list = None):
    if i_range is None:
        i_range = range(start, end + 1)
    for i in i_range:
        fp = f"{folder}/{i:04d}.csv"
        if not os.path.isfile(fp):
            continue
        df = pd.read_csv(fp)
        t = np.arange(len(df)) / 30

        if column is None:
            for col in df.columns:
                plt.plot(t, df[col], label=col)
            plt.legend()
        elif column in df.columns:
            plt.plot(t, df[column])
            plt.title(f"File {i}")
        else:
            continue

        plt.show()


def trim_short_edge_runs(x, k=30):
    valid = ~np.isnan(x)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        raise ValueError("Only NaN values.")

    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    if len(runs[0]) <= k:
        valid[runs[0]] = False
    if len(runs[-1]) <= k:
        valid[runs[-1]] = False
    if not valid.any():
        raise ValueError("Empty array after trimming.")

    return x[np.argmax(valid):len(x) - np.argmax(valid[::-1])]


def preprocess(x: np.ndarray, k_ratio: float = 0.03, low: int = 30,
               high: int = 220) -> np.ndarray:
    k = int(k_ratio * len(x))

    # Jump rules: too large deviations are biologically implausible
    d = x[1:] - x[:-1]  # d_t = x_t - x_{t-1}
    big = d > 20
    small = d < -20
    x[1:][big] = np.nan
    x[:-1][small] = np.nan

    # Remove values outside [30, 220] - biologically implausible
    feasible = ~np.isnan(x) & (low <= x) & (x <= high)
    idx = np.where(feasible)[0]
    if len(idx) == 0:
        raise ValueError("Only NaN or biologically implausible values.")

    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    runs = [r for r in runs if len(r) > k]
    if not runs:
        raise ValueError("Empty array after trimming.")

    first = runs[0][0]
    last = runs[-1][-1] + 1
    x = np.clip(x[first:last], low, high)

    nan = np.isnan(x)
    if not np.any(nan):
        return x

    # Replace missing values by local mean (bandwidth 5)
    mask = (~nan).astype(float)
    vals = np.nan_to_num(x, nan=0.0)

    kernel = np.ones(31)  # kernel length 31 = 1 min at 0.5Hz
    s = np.convolve(vals * mask, kernel, mode="same")
    c = np.convolve(mask, kernel, mode="same")

    fill = nan & (c > 0)
    x[fill] = s[fill] / c[fill]

    return x


def experiment_vital(folder: str, delta_in_s: float, Delta: np.ndarray | float,
                     alpha: float = 0.05,
                     start: int = 1, end: int = 467,
                     column: str = 'Solar8000/ART_MBP') -> dict:
    """
    Main function for running the experiments.

    :param folder: Folder containing MBP csv files.
    :param delta_in_s: Minimal length of stable interval in seconds.
    :param Delta: Scientifically relevant threshold(s).
    :param alpha: Level of test.
    :param start: ID of first recording to consider.
    :param end: ID of last recording to consider.
    :param column: Name of column either 'Solar8000/ART_MBP'
        or 'Solar8000/NIBP_MBP'.
    :return: Dictionary with test results.
    """
    if column not in ['Solar8000/ART_MBP', 'Solar8000/NIBP_MBP']:
        raise ValueError(f"Column {column} is not supported.")

    onsets, optimal_onsets, rejects = [], [], []
    count = 0

    for i in range(start, end + 1):
        fp = f"{folder}/{i:04d}.csv"
        if not os.path.isfile(fp):
            print(f"[skip - file does not exist] {fp}")
            continue
        df = pd.read_csv(fp)

        if column not in df.columns:
            print(f"[skip - {column} not in dataset] {fp}")
            continue

        try:
            data = preprocess(df[column].to_numpy())
        except ValueError:
            continue

        if np.any(np.isnan(data)):
            print(f"[skip - too many missing values] {fp}")
            continue

        n = len(data)
        delta = delta_in_s / (2 * n)  # length of recording at 0.5 Hz

        # Do tests
        baseline = 72.5
        result = stable_state_test(data, Delta, delta, mode='lvl', alpha=alpha,
                                   calibration="both", tune_bw=True, g=baseline)
        onsets.append(result['onset_location'])
        optimal_onsets.append(result['optimal_onset_location'])
        rejects.append(result['reject'])
        count += 1

        if count == 250:
            print(f"Tested {count=} files - finish testing.")
            break
    else:
        print(f"Tested {count=} files only.")

    results = {'onsets': onsets, 'optimal_onsets': optimal_onsets,
               'rejects': rejects}

    return results


if __name__ == '__main__':
    folder_vital = "Add path to raw .vital data"
    folder_csv = "Add path to .csv data"

    # # Convert vital- to csv-file:
    # isolate_mbp(folder_vital, folder_csv, freq=0.5)

    # Display data:
    display_vital(folder_csv, 1, 5, column='Solar8000/ART_MBP')

    # # Conduct experiments:
    # delta_in_s, Delta = 600, 7.5
    # result = experiment_vital(folder_csv, delta_in_s, Delta, end=500)
    # print(result)
