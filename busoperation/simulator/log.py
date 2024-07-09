from typing import List, Optional, Dict, Tuple
from collections import defaultdict

from simulator.virtual_bus import VirtualBus


class StopLog:
    ''' Record the arrival and rtd time of each bus at each stop and corresponding bus schedule deviation (i.e., epsilon)

    Attributes:
        route_arrival_time_seq: a dict mapping route_id to a list of arrival times (before dwelling)
        route_arrival_bus_id_seq: a dict mapping route_id to a list of arrival buses' ids
        route_rtd_time_seq: a dict mapping route_id to a list of rtd (ready-to-depart, after dwelling) times
        route_rtd_bus_id_seq: a dict mapping route_id to a list of rtd buses' ids
        route_bus_epsilon_arrival: a dict mapping route_id to a dict mapping bus_id to arrival schedule deviation
        route_bus_epsilon_rtd: a dict mapping route_id to a dict mapping bus_id to rtd schedule deviation

    Methods:
        record_when_bus_arrival(self, route_id: str, bus_id: str, t: int, epsilon_arrival: float) -> None

    '''

    def __init__(self, stop_id: str, virtual_bus: VirtualBus, has_schedule: bool) -> None:
        self._has_schedule = has_schedule

        # route -> [arrival time at this stop]
        self.route_arrival_time_seq: Dict[str, List[float]] = {}
        # route -> [arrival bus_id]
        self.route_arrival_bus_id_seq: Dict[str, List[str]] = {}

        # route -> [rtd time at this stop]
        self.route_rtd_time_seq: Dict[str, List[float]] = {}
        # route -> [rtd bus_id]
        self.route_rtd_bus_id_seq: Dict[str, List[str]] = {}

        if self._has_schedule:
            # record: route_id -> [bus_id -> epsilon when arrival]
            self.route_bus_epsilon_arrival: Dict[str,
                                                 Dict[str, float]] = defaultdict(dict)
            # record: route_id -> [bus_id -> epsilon when rtd]
            self.route_bus_epsilon_rtd: Dict[str,
                                             Dict[str, float]] = defaultdict(dict)

        # initialization for the virtual bus
        # first arrival and rtd bus is the virutal bus with 'bus_id=0' on each route
        # set its corresponding schedule deviations are 0 if has_schedule is True
        for route_id, stop_arrival_time in virtual_bus.route_stop_arrival_time.items():

            if not stop_id in stop_arrival_time:
                continue

            arrival_time_this_stop = stop_arrival_time[stop_id]
            self.route_arrival_time_seq[route_id] = [arrival_time_this_stop]
            self.route_arrival_bus_id_seq[route_id] = ['0']
            if self._has_schedule:
                self.route_bus_epsilon_arrival[route_id]['0'] = 0

        for route_id, stop_rtd_time in virtual_bus.route_stop_rtd_time.items():

            if not stop_id in stop_rtd_time:
                continue

            rtd_time_this_stop = stop_rtd_time[stop_id]
            self.route_rtd_time_seq[route_id] = [rtd_time_this_stop]
            self.route_rtd_bus_id_seq[route_id] = ['0']
            if self._has_schedule:
                self.route_bus_epsilon_rtd[route_id]['0'] = 0

    def record_when_bus_arrival(self, route_id: str, bus_id: str, t: int, **kwargs) -> None:
        ''' Record the arrival time and arrival schedule deviation of a bus at a stop.

        Schedule deviation is actual arrival time minus scheduled arrival time, which is passed in as `epsilon_arrival`.

        Args:
            route_id: arrival bus's route id
            bus_id: arrival bus's id
            t: arrival time
            kwargs: optional arguments for recording
                - epsilon_arrival: arrival schedule deviation

        '''
        self.route_arrival_time_seq[route_id].append(t)
        self.route_arrival_bus_id_seq[route_id].append(bus_id)

        if 'epsilon_arrival' in kwargs:
            epsilon_arrival = kwargs['epsilon_arrival']
            self.route_bus_epsilon_arrival[route_id][bus_id] = epsilon_arrival

    def record_when_bus_rtd(self, route_id: str, bus_id: str, t: int, **kwargs) -> None:
        ''' Record the rtd time and rtd schedule deviation of a bus at a stop.

        Schedule deviation is actual rtd time minus scheduled rtd time, which is passed in as `epsilon_rtd`.

        Args:
            route_id: rtd bus's route id
            bus_id: rtd bus's id
            t: rtd time
            kwargs: optional arguments for recording
                - epsilon_rtd: rtd schedule deviation

        '''
        self.route_rtd_time_seq[route_id].append(t)
        self.route_rtd_bus_id_seq[route_id].append(bus_id)
        if 'epsilon_rtd' in kwargs:
            epsilon_rtd = kwargs['epsilon_rtd']
            self.route_bus_epsilon_rtd[route_id][bus_id] = epsilon_rtd


class HolderLog:
    def __init__(self, virtual_bus: VirtualBus, has_schedule: bool) -> None:
        self.route_stop_departure_time_seq: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list))
        self.route_stop_departure_bus_id_seq: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list))

        if has_schedule:
            self.route_stop_bus_epsilon_departure: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
                lambda: defaultdict(dict))

        # initialize the departure time of the first virtual bus with `bus_id=0` on each route
        for route_id, stop_departure_time in virtual_bus.route_stop_departure_time.items():
            for stop_id, departure_time in stop_departure_time.items():
                self.route_stop_departure_time_seq[route_id][stop_id] = [
                    departure_time]
                self.route_stop_departure_bus_id_seq[route_id][stop_id] = ['0']

                if has_schedule:
                    # the epsilon_departure of the first virtual bus is 0
                    self.route_stop_bus_epsilon_departure[route_id][stop_id]['0'] = 0

    def record_when_bus_departure(self, stop_id: str, route_id: str,
                                  bus_id: str, t: int, **kwargs) -> None:
        self.route_stop_departure_time_seq[route_id][stop_id].append(t)
        self.route_stop_departure_bus_id_seq[route_id][stop_id].append(bus_id)

        if 'epsilon_departure' in kwargs:
            epsilon_departure = kwargs['epsilon_departure']
            self.route_stop_bus_epsilon_departure[route_id][stop_id][bus_id] = epsilon_departure


class BusRunningLog:
    def __init__(self, schedule_headway: float, virtual_bus_stop_arrival_time: Dict[str, float],
                 virtual_bus_stop_rtd_time: Dict[str, float], virtual_bus_stop_departure_time: Dict[str, float]) -> None:

        self.virtual_bus_stop_arrival_time = virtual_bus_stop_arrival_time
        self.virtual_bus_stop_rtd_time = virtual_bus_stop_rtd_time
        self.virtual_bus_stop_departure_time = virtual_bus_stop_departure_time

        # mean of H
        self.schedule_headway = schedule_headway
        # record the dispatch time from the terminal
        self.dispatch_time: Optional[int] = None
        # record the end time of the bus operation
        self.end_time: Optional[int] = None

        # record the link travel time deviation from mean at each stop
        self.link_tt_deviation: Dict[str, float] = {}
        # record the dwell time at each stop
        self.stop_dwell_time: Dict[str, int] = defaultdict(int)
        # record visited stops
        self.visited_stops: List[str] = []

        # record the schedule deviation when arrival at each stop
        self.stop_epsilon_arrival: Dict[str, float] = {}
        # record the schedule deviation when ready-to-departure at each stop
        self.stop_epsilon_rtd: Dict[str, float] = {}
        # record the schedule deviation when departure after holding (at the holder)
        self.stop_epsilon_departure: Dict[str, float] = {}

        # record the arrival time at each stop
        self.stop_arrival_time: Dict[str, int] = {}
        self.stop_rtd_time: Dict[str, int] = {}
        self.stop_departure_time: Dict[str, int] = {}

        # record the queueing delay at each stop
        self.stop_queueing_delay: Dict[str, int] = defaultdict(int)

    def record_when_dispatch(self, t: int) -> None:
        ''' Record the dispatch time from the terminal.

        '''
        self.dispatch_time = t

    def record_when_enter_link(self, link_id: str, tt_deviation: float) -> None:
        self.link_tt_deviation[link_id] = tt_deviation

    def record_when_arrival(self, stop_id: str, t: int,  **kwargs) -> Optional[float]:
        ''' Record the arrival time and schedule deviation of this bus at a stop.

        Args:
            stop_id: arrived stop id
            t: arrival time
            last_arrival_idx_count: the count of arrivals at this stop

        '''

        self.stop_arrival_time[stop_id] = t

        epsilon_arrival = None
        if 'last_arrival_idx_count' in kwargs:
            last_arrival_idx_count = kwargs['last_arrival_idx_count']
            # get the virtual bus arrival time at this stop
            virtual_bus_arrival_time = self.virtual_bus_stop_arrival_time[stop_id]

            # calculate how many shifts from the virtual bus arrival time
            shift = self.schedule_headway * last_arrival_idx_count

            schedule_arrival = virtual_bus_arrival_time + shift
            epsilon_arrival = t - schedule_arrival
            self.stop_epsilon_arrival[stop_id] = epsilon_arrival

        return epsilon_arrival

    def record_when_dwell(self, stop_id: str) -> None:
        self.stop_dwell_time[stop_id] += 1

    def record_when_rtd(self, stop_id: str, t: int, **kwargs) -> Optional[float]:
        self.stop_rtd_time[stop_id] = t
        self.visited_stops.append(stop_id)

        epsilon_rtd = None
        if 'last_rtd_idx_count' in kwargs:
            last_rtd_idx_count = kwargs['last_rtd_idx_count']
            # get the virtual bus rtd time at this stop
            virtual_bus_rtd_time = self.virtual_bus_stop_rtd_time[stop_id]

            # calculate how many shifts from the virtual bus rtd time
            shift = self.schedule_headway * last_rtd_idx_count

            schedule_rtd = virtual_bus_rtd_time + shift
            epsilon_rtd = t - schedule_rtd
            self.stop_epsilon_rtd[stop_id] = epsilon_rtd
        return epsilon_rtd

    def record_when_departure(self, stop_id: str, t: int, **kwargs) -> Optional[float]:
        self.stop_departure_time[stop_id] = t

        epsilon_departure = None
        if 'last_departure_idx_count' in kwargs:
            last_departure_idx_count = kwargs['last_departure_idx_count']
            # get the virtual bus departure time at this stop
            virtual_bus_departure_time = self.virtual_bus_stop_departure_time[stop_id]

            # calculate how many shifts from the virtual bus departure time
            shift = self.schedule_headway * last_departure_idx_count

            schedule_departure = virtual_bus_departure_time + shift
            epsilon_departure = t - schedule_departure
            self.stop_epsilon_departure[stop_id] = epsilon_departure

        return epsilon_departure

    def record_when_finish(self, t: int) -> None:
        self.end_time = t

    def record_when_queue(self, stop_id: str) -> None:
        self.stop_queueing_delay[stop_id] += 1
