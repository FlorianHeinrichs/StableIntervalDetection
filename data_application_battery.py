import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from estimators import local_linear_estimator
from experiments import stable_state_test


def display_battery(folder: str):
    temp_dates = [(0, "20120618"), (25, "20120905"), (50, "20120702")]

    fig, axes = plt.subplots(1, 2)

    for i, ax in zip([7, 8], axes):
        for temp, date in temp_dates:
            fp = f"{folder}A1-00{i}-OCV-{temp}-{date}.xlsx"
            df = pd.read_excel(fp, sheet_name="Sheet1", header=0, usecols="A:C")

            start, end = np.argwhere(df.iloc[:, 1].to_numpy() < 0)[[0, -1]][:, 0]
            t = df.iloc[start: end + 1, 0] / 60
            v = df.iloc[start: end + 1, 2]
            ax.plot(t, v)
    plt.show()


def experiment_battery(folder: str, delta_in_h: float,
                       Delta_per_h: np.ndarray | float, alpha: float = 0.05,
                       bw: int = 60):
    """
    Main function for running the experiments.

    :param folder: Folder containing battery data.
    :param delta_in_h: Minimal length of stable interval in hours.
    :param Delta_per_h: Scientifically relevant threshold(s) as slope in V/h.
    :param alpha: Level of test.
        or 'Solar8000/NIBP_MBP'.
    :param bw: Bandwidth used for local linear estimation.
    :return: Dictionary with test results.
    """
    temp_dates = [(0, "20120618"), (25, "20120905"), (50, "20120702")]

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    for i in [7, 8]:
        for temp, date in temp_dates:
            fp = f"{folder}A1-00{i}-OCV-{temp}-{date}.xlsx"
            df = pd.read_excel(fp, sheet_name="Sheet1", header=0, usecols="A:C")

            start, end = np.argwhere(df.iloc[:, 1].to_numpy() < 0)[[0, -1]][:, 0]
            t_in_h = df.iloc[start: end + 1, 0].to_numpy() / 3600
            n_h = t_in_h[-1]
            v = df.iloc[start: end + 1, 2].to_numpy()

            delta = delta_in_h / n_h  # length of recording at 1 Hz
            Delta = Delta_per_h * n_h

            # Do tests
            result = stable_state_test(v, Delta, delta, mode='drv', alpha=alpha,
                                       calibration="both", bw=bw)
            onset = result['onset_location']
            optimal_onset = result['optimal_onset_location']
            p_values = result['p_value']

            result_str = (f"Battery {i} at temperature {temp}°C and delta "
                          f"{delta_in_h}: {p_values}")
            print(result_str)

            if i == 7 and temp == 25 and delta_in_h == 2:
                fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
                ax1, ax2 = axes
                ax1.plot(t_in_h, v, color="black", lw=2.0)

                _, d = local_linear_estimator(v, bw)
                ax2.plot(t_in_h, d / n_h, color="black", lw=2.0)
                ax2.axhline(-Delta_per_h, color="tab:red", lw=1.4, ls="--")
                ax2.axhline(Delta_per_h, color="tab:red", lw=1.4, ls="--")
                ax2.fill_between(t_in_h, -Delta_per_h, Delta_per_h, color="tab:red", alpha=0.08)
                ax2.set_xlim([0, n_h])
                ax2.set_ylim([-0.08, 0.02])

                ax2.axvspan(onset * n_h, (onset * n_h) + delta_in_h, color="tab:green",
                           alpha=0.18)
                ax2.axvspan(optimal_onset * n_h, (optimal_onset * n_h) + delta_in_h,
                           color="tab:blue", alpha=0.18)

                fig.tight_layout()
                plt.show()



if __name__ == '__main__':
    folder = "Add path to folder containin battery data"
    files = [folder + f"A1-00{i}-OCV-25-20120905.xlsx" for i in [7, 8]]

    # display_battery(folder)

    delta_in_h, Delta_per_h = 8, 0.01

    bw_in_min = 30
    bw = bw_in_min * 12

    for delta_in_h in range(1, 9):
        experiment_battery(folder, delta_in_h, Delta_per_h, bw=bw)
