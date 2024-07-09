from typing import Literal
from collections import defaultdict

import matplotlib.pyplot as plt

from dataclasses import dataclass


@dataclass
class TrajectoryPoint:
    spot_type: str
    spot_id: str
    distance_from_terminal: float
    status: Literal['running_on_link', 'queueing_at_stop',
                    'dwelling_at_stop', 'holding', 'finished']


def plot_time_space_diagram(buses):
    _, ax = plt.subplots()
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('Offset (m)', fontsize=12)

    for bus in buses:
        # if not bus.route_id == 'B2A':
        #     continue

        # plot trajectory
        x = []
        y = []
        for t, point in bus.trajectory.items():
            x.append(t)
            y.append(point.distance_from_terminal)
        ax.plot(x, y, 'k')

        # plot holding durations
        hold_xs = defaultdict(list)
        hold_ys = {}
        for t, point in bus.trajectory.items():
            if point.spot_type == 'holder':
                hold_xs[point.spot_id].append(t)
                hold_ys[point.spot_id] = point.distance_from_terminal
        for spot_id, xs in hold_xs.items():
            start, end = min(xs), max(xs)
            y = hold_ys[spot_id]
            ax.hlines(y=y, xmin=start, xmax=end,
                      color='green', linewidth=3.0)

        # plot queueing durations
        queue_xs = defaultdict(list)
        queue_ys = {}
        for t, point in bus.trajectory.items():
            if point.status == 'queueing_at_stop':
                queue_xs[point.spot_id].append(t)
                queue_ys[point.spot_id] = point.distance_from_terminal

        for spot_id, xs in queue_xs.items():
            start, end = min(xs), max(xs)
            y = queue_ys[spot_id]
            ax.hlines(y=y, xmin=start, xmax=end,
                      color='red', linewidth=3.0)

    plt.show()
