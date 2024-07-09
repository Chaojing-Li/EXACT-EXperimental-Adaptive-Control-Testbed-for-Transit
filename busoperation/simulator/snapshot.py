from dataclasses import dataclass, field
from typing import List, Literal, Dict, Tuple, Optional


@dataclass(frozen=True)
class BusSnapshot:
    ''' The snapshot of a bus at a certain time.

    Attributes:
        bus_id: a string that identifies the bus on a route.
        route_id: a string that identifies the route.
        is_need_to_hold: a boolean indicating whether the bus needs to be held at the holder.
        pax_num: the number of passengers currently on the bus.
        loc_relative_to_terminal: the location of the bus relative to the terminal.
        status: the status of the bus:
            - dispatching: the bus is dispatching from the terminal (last 1 second 
                as it will be running on the first link in the next second).
            - running_on_link: the bus is running on a link.
            - decelerating: the bus is decelerating to enter a stop.
            - queueing_at_stop: the bus is in the entry queue of a stop.
            - dwelling_at_stop: the bus is dwelling at a stop.
            - accelerating: the bus is accelerating to exit a stop.
            - holding: the bus is being held in a holder.
            - finished: the bus has finished its service, i.e., reached the end terminal.
        stop_dwell_time: the dwell time at each stop. for the newest stop, it may still be updating.
        link_tt_deviation: the travel time deviation from mean on each link, recorded when entering the link.
        stop_epsilon_arrival: the schedule deviation when arrival at each stop.
        stop_epsilon_rtd: the schedule deviation when ready-to-departure at each stop.
        stop_epsilon_departure: the schedule deviation when departure after holding (at the holder).
        visited_stops: the list of stops that the bus has visited.

    '''
    bus_id: str
    route_id: str
    is_need_to_hold: bool
    pax_num: int
    loc_relative_to_terminal: float
    status: Literal['running_on_link', 'queueing_at_stop',
                    'dwelling_at_stop', 'holding', 'finished']
    stop_dwell_time: Dict[str, int]
    link_tt_deviation: Dict[str, float]
    stop_epsilon_arrival: Dict[str, float]
    stop_epsilon_rtd: Dict[str, float]
    stop_epsilon_departure: Dict[str, float]
    visited_stops: List[str]


@dataclass(frozen=True)
class StopSnapshot:
    ''' The snapshot of a stop at a certain time.

        Note that the word snapshot might be a little misleading 
        because it actually stores all the historical information of the stop.

    Attributes:
        stop_id: a string that identifies the stop.
        pax_num: the number of passengers currently waiting at the stop.
        route_arrival_time_seq: the arrival time sequence of buses on each route
            {route_id: [arrival_time_1, arrival_time_2, ...]}.
        route_arrival_bus_id_seq: the bus id sequence of buses on each route
            {route_id: [bus_id_1, bus_id_2, ...]}.
        route_rtd_time_seq: the ready-to-departure time sequence of buses on each route
            {route_id: [rtd_time_1, rtd_time_2, ...]}.
        route_rtd_bus_id_seq: the bus id sequence of buses on each route
            {route_id: [bus_id_1, bus_id_2, ...]}.
        route_bus_epsilon_arrival: the schedule deviation when arrival for each bus on each route
            {route_id: {bus_id: epsilon_arrival}}.
        route_bus_epsilon_rtd: the schedule deviation when ready-to-departure for each bus on each route
            {route_id: {bus_id: epsilon_rtd}}.

    '''
    stop_id: str
    pax_num: int
    route_arrival_time_seq: Dict[str, List[float]]
    route_arrival_bus_id_seq: Dict[str, List[str]]
    route_rtd_time_seq: Dict[str, List[float]]
    route_rtd_bus_id_seq: Dict[str, List[str]]
    route_bus_epsilon_arrival: Optional[Dict[str, Dict[str, float]]] = None
    route_bus_epsilon_rtd: Optional[Dict[str, Dict[str, float]]] = None


@dataclass(frozen=True)
class HolderSnapshot:
    ''' The snapshot of a holder at a certain time.

    Attributes:
        action_buses: the list of buses that need to be determined a hold time at the holder
            [(stop_id_1, route_id_1, bus_id_1), (stop_id_2, route_id_2, bus_id_2), ...].
        route_stop_departure_time_seq: the departure time sequence of buses at each stop on each route
            {route_id: {stop_id: [departure_time_1, departure_time_2, ...]}}.
        route_stop_departure_bus_id_seq: the bus id sequence of buses at each stop on each route
            {route_id: {stop_id: [bus_id_1, bus_id_2, ...]}}.
        route_stop_bus_epsilon_departure: the schedule deviation when departure for each bus at holder (of each stop) on each route
            {route_id: {stop_id: {bus_id: epsilon_departure}}}.

    '''
    action_buses: List[Tuple[str, str, str]]
    route_stop_departure_time_seq: Dict[str, Dict[str, List[float]]]
    route_stop_departure_bus_id_seq: Dict[str, Dict[str, List[str]]]
    route_stop_bus_epsilon_departure: Optional[Dict[str,
                                                    Dict[str, Dict[str, float]]]] = None


@dataclass
class Snapshot:
    ''' The snapshot of the whole system at a certain time.

    Attributes:
        t: the current time.
        bus_snapshots: the snapshot of all buses, {(route_id, bus_id): BusSnapshot}
        stop_snapshots: the snapshot of all stops.
        holder_snapshot: the snapshot of the holder.
        action_record: the record of holding time of each bus at each stop

    '''
    t: int
    bus_snapshots: Dict[Tuple[str, str], BusSnapshot]
    stop_snapshots: Dict[str, StopSnapshot]
    holder_snapshot: HolderSnapshot
    action_record: Dict[Tuple[str, str, str],
                        float] = field(default_factory=lambda: {})

    def get_holder_epsilon(self, node_id: str, route_id: str, bus_id: str) -> float:
        ''' Get the schedule deviation when departure for the bus at the holder (of the `node_id`) on the `route_id`

        '''
        stop_epsilon_departure = self.bus_snapshots[(
            route_id, bus_id)].stop_epsilon_departure
        return stop_epsilon_departure[node_id]

    def get_stop_epsilon(self, route_id: str, curr_stop_id: str, curr_bus_id: str, query_bus='last'):
        ''' For the current stop with `curr_stop_id` on the `route_id`, get the epsilon_arrival and epsilon_rtd of the `[query_bus]`
        '''
        # find the bus id of the last arrival bus
        arrival_bus_id_seq = self.stop_snapshots[curr_stop_id].route_arrival_bus_id_seq[route_id]
        arrival_time_seq = self.stop_snapshots[curr_stop_id].route_arrival_time_seq[route_id]
        curr_bus_index = arrival_bus_id_seq.index(curr_bus_id)
        last_arrival_bus_id = arrival_bus_id_seq[curr_bus_index - 1]

        # find the bus id of the last rtd bus
        rtd_bus_id_seq = self.stop_snapshots[curr_stop_id].route_rtd_bus_id_seq[route_id]
        curr_bus_index = rtd_bus_id_seq.index(curr_bus_id)
        last_rtd_bus_id = rtd_bus_id_seq[curr_bus_index - 1]

        epsilon_arrival = self.stop_snapshots[curr_stop_id].route_bus_epsilon_arrival[route_id][last_arrival_bus_id]
        epsilon_rtd = self.stop_snapshots[curr_stop_id].route_bus_epsilon_rtd[route_id][last_rtd_bus_id]

        return epsilon_arrival, epsilon_rtd

    def get_bus_epsilon(self, route_id: str, curr_bus_id: str, query_stop_id: str):
        ''' For the current bus with `curr_bus_id` on the `route_id`, get the epsilon_arrival and epsilon_rtd of the `query_stop_id`

        Args:
            route_id: route id
            curr_bus_id: current querying bus id
            query_stop_id: the stop id to query

        Returns:
            epsilon_arrival: arrival schedule deviation for the current bus at the `query_stop_id`
            epsilon_rtd: ready-to-depart schedule deviation of the current bus at the `query_stop_id`

        '''

        stop_epsilon_arrival = self.bus_snapshots[(
            route_id, curr_bus_id)].stop_epsilon_arrival
        epsilon_arrival = stop_epsilon_arrival[query_stop_id]

        stop_epsilon_rtd = self.bus_snapshots[(
            route_id, curr_bus_id)].stop_epsilon_rtd
        epsilon_rtd = stop_epsilon_rtd[query_stop_id]

        return epsilon_arrival, epsilon_rtd

    def get_last_rtd_time(self, route_id: str, stop_id: str) -> float:
        rtd_times = self.stop_snapshots[stop_id].route_rtd_time_seq[route_id]
        return rtd_times[-2]

    def record_holding_time(self, stop_bus_hold_time: Dict[Tuple[str, str, str], float]) -> None:
        ''' Record the holding time of each bus at each stop 

        Args:
            stop_bus_hold_time: the holding time of each bus at each stop
                {(stop_id, route_id, bus_id): holding_time}

        '''

        for (stop_id, route_id, bus_id), holding_time in stop_bus_hold_time.items():
            self.action_record[(stop_id, route_id, bus_id)] = holding_time
