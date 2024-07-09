from typing import Dict, List
import numpy as np


def calculate_headway_std(times: List[float]) -> float:
    times.sort()
    times_array = np.array(times, dtype=np.int32)
    headways = np.diff(times_array).tolist()
    headway_std = float(np.std(headways))
    return headway_std


def calculate_mean_abs_epsilon(epsilons: List[float]) -> float:
    return float(np.mean(np.abs(np.array(epsilons))))
