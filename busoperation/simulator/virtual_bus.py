from collections import defaultdict
from typing import Dict
from copy import deepcopy

from setup.blueprint import Blueprint


class VirtualBus:
    ''' Virtual bus that determines passenger arrival start time at each stop.

    Properties:
        route_stop_arrival_time: the arrival time of the virtual bus at each stop
        route_stop_rtd_time: the ready-to-departure time of the virtual bus at each stop
        route_stop_departure_time: the departure time of the virtual bus at each stop
        route_stop_pax_arrival_start_time: the passenger arrival start time at each stop

    Methods:
        initialize_with_perfect_schedule(self, route_stop_arrival_rate: Dict[str, Dict[str, float]], slack: float)
        initialize_by_data(self, route_stop_rtd_time: Dict[str, Dict[str, float]])
        update_trajectory(self, route_stop_average_hold_time: Dict[str, Dict[str, float]])

    '''

    def __init__(self, blueprint: Blueprint) -> None:
        self._blueprint = blueprint
        self._route_stop_arrival_time = defaultdict(dict)
        self._route_stop_rtd_time = defaultdict(dict)
        self._route_stop_departure_time = defaultdict(dict)

    @property
    def route_stop_arrival_time(self) -> Dict[str, Dict[str, float]]:
        return dict(self._route_stop_arrival_time)

    @property
    def route_stop_rtd_time(self) -> Dict[str, Dict[str, float]]:
        return dict(self._route_stop_rtd_time)

    @property
    def route_stop_departure_time(self) -> Dict[str, Dict[str, float]]:
        return dict(self._route_stop_departure_time)

    @property
    def route_stop_pax_arrival_start_time(self) -> Dict[str, Dict[str, float]]:
        return dict(self._route_stop_rtd_time)

    def initialize_with_perfect_schedule(self, route_stop_arrival_rate: Dict[str, Dict[str, float]], slack: float):
        ''' Initialize the virtual bus using the perfect schedule.

        The virtual bus starts from the terminal at time 0, follows the mean travel time and boarding time, 
        and holds at each stop for the given slack; i.e., the virtual bus adheres perfectly to the schedule.

        Alighting is not considered in this case, i.e., boarding dominates alighting.

        '''
        self._route_stop_arrival_rate = route_stop_arrival_rate
        route_stop_average_hold_time = self._initialize_average_hold_time(
            slack)
        self.update_trajectory(route_stop_average_hold_time)

    def initialize_by_data(self, route_stop_rtd_time: Dict[str, Dict[str, float]]):
        ''' Initialize the virtual bus by the given route_stop_rtd_time.

        Historical data has no holding, so departure time is the same as arrival time.

        '''
        self._route_stop_arrival_time = deepcopy(route_stop_rtd_time)
        self._route_stop_rtd_time = deepcopy(route_stop_rtd_time)
        self._route_stop_departure_time = deepcopy(route_stop_rtd_time)

    def update_trajectory(self, route_stop_average_hold_time: Dict[str, Dict[str, float]]):
        ''' Given the average holding time at each stop, update the virtual bus schedule.

        Assume that the virtual bus adhere the perfect schedule

        '''
        for route_id, route in self._blueprint.route_schema.route_details_by_id.items():
            visit_seq_nodes = [route.terminal_id] + route.visit_seq_stops
            H = route.schedule_headway
            stop_boarding_rate = route.boarding_rate
            t = 0
            # set the arrival, rtd and departure time of the virtual bus at the terminal all to be 0
            self._route_stop_arrival_time[route_id][route.terminal_id] = t
            self._route_stop_rtd_time[route_id][route.terminal_id] = t
            self._route_stop_departure_time[route_id][route.terminal_id] = t

            # 'terminal' -> 'link-0' -> 'stop-0' -> 'link-1' -> 'stop-1' -> 'link-2' -> ... -> 'ending terminal'
            for head_node, tail_node in zip(visit_seq_nodes[:-1], visit_seq_nodes[1:]):
                # in chronological order, the departed bus will:
                # (1) travel to the next stop;
                # (2) serve the passengers at the next stop, then ready for departure;
                # (3) be held in the holder

                # (1) link travel time from node s to node s+1
                link_id = self._blueprint.get_next_link_id(route_id, head_node)
                link_time = self._blueprint.network.link_distribution[link_id].tt_mean
                t += link_time
                self._route_stop_arrival_time[route_id][tail_node] = t

                # (2) boarding time at node s+1
                arrival_rate = self._route_stop_arrival_rate[route_id][tail_node]
                board_rate = stop_boarding_rate[tail_node]
                boarding_time = arrival_rate / board_rate * H
                t += boarding_time
                self._route_stop_rtd_time[route_id][tail_node] = t

                # (3) holding time at node s+1
                hold_time = route_stop_average_hold_time[route_id][tail_node]
                t += hold_time
                self._route_stop_departure_time[route_id][tail_node] = t

    def _initialize_average_hold_time(self, slack) -> Dict[str, Dict[str, float]]:
        route_stop_average_hold_time = defaultdict(dict)
        for route_id, route in self._blueprint.route_schema.route_details_by_id.items():
            for stop_id in route.visit_seq_stops:
                route_stop_average_hold_time[route_id][stop_id] = slack
        return dict(route_stop_average_hold_time)
