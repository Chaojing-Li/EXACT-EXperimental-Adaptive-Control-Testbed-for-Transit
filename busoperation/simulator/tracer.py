from collections import defaultdict
from typing import Dict, List, Tuple, Literal

import numpy as np

from .bus import Bus
from .holder import Holder
from .link import Link
from .pax import Pax
from .snapshot import Snapshot, StopSnapshot, BusSnapshot
from .stop import Stop
from .utils import calculate_headway_std, calculate_mean_abs_epsilon


class Tracer:
    ''' The tracer that traces the state of the system at each time step.

    Methods:
        take_snapshot(self, t: int, links: Dict[str, Link], stops: Dict[str, Stop], holder: Holder) -> Snapshot
        get_metric(self, route_stop_ids: Dict[str, List[str]], 
                    total_buses: List[Bus], 
                    left_paxs: List[Pax], 
                    metric_names: List[Literal['headway_std', 'schedule_deviation', 
                    'pax_in_vehicle_wait_time', 'pax_out_vehicle_wait_time', 
                    'hold_time', 'queueing_delay']]) -> Dict
    '''

    def __init__(self) -> None:
        self._snapshots: List[Snapshot] = []

    def take_snapshot(self, t: int, links: Dict[str, Link], stops: Dict[str, Stop], holder: Holder) -> Snapshot:
        ''' Take a snapshot of the system at each time t.

        Args:
            t: the current time step
            links: all the links in the system
            stops: all the stops in the system
            holder: the holder that holds the buses at stops

        Returns:
            snapshot: the snapshot of the system at time t

        '''
        bus_snapshots: Dict[Tuple[str, str], BusSnapshot] = {}
        stop_snapshots: Dict[str, StopSnapshot] = {}

        # links
        for link in links.values():
            for bus in link.buses:
                bus_snapshot = bus.take_snapshot()
                bus_snapshots[(bus.route_id, bus.bus_id)] = bus_snapshot

        # stops
        for stop_id, stop in stops.items():
            stop_snapshots[stop_id] = stop.take_snapshot()

            for bus in stop.get_total_buses():
                bus_snapshot = bus.take_snapshot()
                bus_snapshots[(bus.route_id, bus.bus_id)] = bus_snapshot

        # holder
        for (stop_id, route_id, bus_id), bus in holder.stop_identifier_bus.items():
            bus_snapshot = bus.take_snapshot()
            bus_snapshots[(bus.route_id, bus.bus_id)] = bus_snapshot

        holder_snapshot = holder.take_snapshot()
        snapshot = Snapshot(t, bus_snapshots, stop_snapshots, holder_snapshot)

        self._snapshots.append(snapshot)
        return snapshot

    def get_metric(self, route_stop_ids: Dict[str, List[str]],
                   total_buses: List[Bus],
                   left_paxs: List[Pax],
                   metric_names: List[Literal['headway_std', 'schedule_deviation',
                                              'pax_in_vehicle_wait_time', 'pax_out_vehicle_wait_time',
                                              'hold_time', 'queueing_delay']]) -> Dict:
        ''' Get the metrics of given stops of each route for a single episode.

        Args:
            route_stop_ids: the stop ids that need to be counted for each route
            total_buses: all the buses created in the simulation
            left_paxs: the passengers that have finished their trips
            metric_names: the metrics that need to be calculated

        Returns:
            metrics: a dictionary of metrics with key as the metric name and value as the metric value

        '''
        metrics = {}
        # a bus can be identified by its route_id and bus_id
        counted_bus: List[Tuple[str, str]] = []

        # 1. stop-level metrics
        for route_id, stop_ids in route_stop_ids.items():
            stop_arrival_time_seq: Dict[str, List[float]] = defaultdict(list)
            stop_rtd_time_seq: Dict[str, List[float]] = defaultdict(list)
            stop_departure_time_seq: Dict[str, List[float]] = defaultdict(list)

            if 'schedule_deviation' in metric_names:
                stop_arrival_epsilon_seq: Dict[str,
                                               List[float]] = defaultdict(list)
                stop_rtd_epsilon_seq: Dict[str,
                                           List[float]] = defaultdict(list)
                stop_departure_epsilon_seq: Dict[str,
                                                 List[float]] = defaultdict(list)

            for stop_id in stop_ids:
                for bus in total_buses:
                    # the unheld bus treated as initial condition and should not be counted
                    if not bus.is_need_to_hold:
                        continue
                    counted_bus.append((bus.route_id, bus.bus_id))

                    if bus.route_id == route_id and stop_id in bus.bus_log.stop_arrival_time:
                        stop_arrival_time_seq[stop_id].append(
                            bus.bus_log.stop_arrival_time[stop_id])

                        if 'schedule_deviation' in metric_names:
                            stop_arrival_epsilon_seq[stop_id].append(
                                bus.bus_log.stop_epsilon_arrival[stop_id])

                    if bus.route_id == route_id and stop_id in bus.bus_log.stop_rtd_time:
                        stop_rtd_time_seq[stop_id].append(
                            bus.bus_log.stop_rtd_time[stop_id])

                        if 'schedule_deviation' in metric_names:
                            stop_rtd_epsilon_seq[stop_id].append(
                                bus.bus_log.stop_epsilon_rtd[stop_id])

                    if bus.route_id == route_id and stop_id in bus.bus_log.stop_departure_time:
                        stop_departure_time_seq[stop_id].append(
                            bus.bus_log.stop_departure_time[stop_id])

                        if 'schedule_deviation' in metric_names:
                            stop_departure_epsilon_seq[stop_id].append(
                                bus.bus_log.stop_epsilon_departure[stop_id])

            arrival_headway_stds = []
            rtd_headway_stds = []
            departure_headway_stds = []
            if 'schedule_deviation' in metric_names:
                epsilon_arrival_mean_abs = []
                epsilon_rtd_mean_abs = []
                epsilon_departure_mean_abs = []

            for stop_id, arrival_time_seq in stop_arrival_time_seq.items():
                arrival_headway_std = calculate_headway_std(arrival_time_seq)
                arrival_headway_stds.append(arrival_headway_std)

            for stop_id, rtd_time_seq in stop_rtd_time_seq.items():
                rtd_headway_std = calculate_headway_std(rtd_time_seq)
                rtd_headway_stds.append(rtd_headway_std)

            for stop_id, departure_time_seq in stop_departure_time_seq.items():
                departure_headway_std = calculate_headway_std(
                    departure_time_seq)
                departure_headway_stds.append(departure_headway_std)

            if 'schedule_deviation' in metric_names:
                for stop_id, epsilon_arrival_seq in stop_arrival_epsilon_seq.items():
                    mean_abs_epsilon_arrival = calculate_mean_abs_epsilon(
                        epsilon_arrival_seq)
                    epsilon_arrival_mean_abs.append(mean_abs_epsilon_arrival)

                for stop_id, epsilon_rtd_seq in stop_rtd_epsilon_seq.items():
                    mean_abs_epsilon_rtd = calculate_mean_abs_epsilon(
                        epsilon_rtd_seq)
                    epsilon_rtd_mean_abs.append(mean_abs_epsilon_rtd)

                for stop_id, epsilon_departure_seq in stop_departure_epsilon_seq.items():
                    mean_abs_epsilon_departure = calculate_mean_abs_epsilon(
                        epsilon_departure_seq)
                    epsilon_departure_mean_abs.append(
                        mean_abs_epsilon_departure)

            if 'headway_std' in metric_names:
                metrics[f'route-{route_id}\'s arrival_headway_std'] = np.mean(
                    arrival_headway_stds)
                metrics[f'route-{route_id}\'s rtd_headway_std'] = np.mean(
                    rtd_headway_stds)
                metrics[f'route-{route_id}\'s departure_headway_std'] = np.mean(
                    departure_headway_stds)

            if 'schedule_deviation' in metric_names:
                metrics[f'route-{route_id}\'s arrival_schedule_deviation'] = np.mean(
                    epsilon_arrival_mean_abs)
                metrics[f'route-{route_id}\'s rtd_schedule_deviation'] = np.mean(
                    epsilon_rtd_mean_abs)
                metrics[f'route-{route_id}\'s departure_schedule_deviation'] = np.mean(
                    epsilon_departure_mean_abs)

        # 2. bus-level metrics
        if 'hold_time' in metric_names:
            # key is the route id and the value is a list of total queueing delay (sum of all the stops) for each bus
            route_bus_total_hold_time: Dict[str,
                                            List[float]] = defaultdict(list)

            route_bus_stop_hold_times: Dict[str, Dict[str, List[float]]] = defaultdict(
                lambda: defaultdict(list))
            for snapshot in self._snapshots:
                for (stop_id, route_id, bus_id), holding_time in snapshot.action_record.items():
                    if (route_id, bus_id) not in counted_bus:
                        continue
                    route_bus_stop_hold_times[route_id][bus_id].append(
                        holding_time)

            for route_id, bus_stop_hold_times in route_bus_stop_hold_times.items():
                for bus_id, stop_hold_times in bus_stop_hold_times.items():
                    route_bus_total_hold_time[route_id].append(
                        sum(stop_hold_times))

            total_hold_times = []
            for route_id, hold_times in route_bus_total_hold_time.items():
                bus_mean_hold_time = np.mean(hold_times)
                metrics[f'route-{route_id}\'s holding time'] = bus_mean_hold_time
                total_hold_times.append(bus_mean_hold_time)
            metrics[f'total average bus holding time'] = np.mean(
                total_hold_times)

        if 'queueing_delay' in metric_names:
            route_queueing_delays: Dict[str, List[float]] = defaultdict(list)
            for route_id, stop_ids in route_stop_ids.items():
                bus_queueing_delays = []
                for bus in total_buses:
                    if not bus.is_need_to_hold or bus.route_id != route_id:
                        continue

                    queue_delay_sum = 0
                    for stop_id in stop_ids:
                        queue_delay = bus.bus_log.stop_queueing_delay[stop_id]
                        queue_delay_sum += queue_delay
                    bus_queueing_delays.append(queue_delay_sum)

                route_queueing_delays[route_id] = bus_queueing_delays

            mean_queue_delay = np.mean(
                [np.mean(queueing_delays) for queueing_delays in route_queueing_delays.values()])
            metrics['queueing_delay'] = mean_queue_delay

        if 'pax_in_vehicle_wait_time' in metric_names:
            # in_vehicle_wait_time = [pax.in_vehicle_delay for pax in left_paxs]
            in_vehicle_wait_time = [
                pax.alight_time - pax.board_time for pax in left_paxs
                if pax.board_time is not None and pax.alight_time is not None]
            metrics['pax_in_vehicle_wait_time'] = np.mean(in_vehicle_wait_time)

        if 'pax_out_vehicle_wait_time' in metric_names:
            # out_vehicle_wait_time = [
            #     pax.out_vehicle_delay for pax in left_paxs]
            out_vehicle_wait_time = [
                pax.board_time - pax.arrival_time for pax in left_paxs if pax.board_time is not None]
            metrics['pax_out_vehicle_wait_time'] = np.mean(
                out_vehicle_wait_time)

        return metrics

    def get_stop_average_hold_time(self) -> Dict[str, Dict[str, float]]:
        ''' Get the average holding time of each stop for each route.

        Returns:
            route_stop_hold_time: a dictionary {route_id -> {stop_id -> average_hold_time}}

        '''
        route_stop_hold_times: Dict[Tuple[str, str], List] = defaultdict(list)
        for snapshot in self._snapshots:
            for (stop_id, route_id, bus_id), holding_time in snapshot.action_record.items():
                route_stop_hold_times[(route_id, stop_id)].append(holding_time)

        route_stop_hold_time = defaultdict(dict)
        for (route_id, stop_id), holding_times in route_stop_hold_times.items():
            route_stop_hold_time[route_id][stop_id] = np.mean(holding_times)
        return dict(route_stop_hold_time)
