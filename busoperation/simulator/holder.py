from collections import defaultdict
from typing import Dict, List, Tuple, Optional, DefaultDict

from agent.agent import Agent
from simulator.virtual_bus import VirtualBus

from .bus import Bus
from .snapshot import HolderSnapshot
from .log import HolderLog


class Holder:
    ''' A unified holder for managing holdings at all stops.

    Attributes:
        _identifier_bus: a dictionary with key as (stop_id, route_id, bus_id) and value as Bus
        _identifier_time: a dictionary with key as (stop_id, route_id, bus_id) and value as dynamic remaining hold time
        holder_log: a HolderLog object for logging

    '''

    _identifier_bus: DefaultDict[Tuple[str, str, str], Bus]
    _identifier_time: DefaultDict[Tuple[str, str, str], Optional[float]]
    log: HolderLog

    def __init__(self, agent: Agent, virtual_bus: VirtualBus, has_schedule: bool) -> None:
        # Dict[Tuple[released stop id, route id, bus id], Bus]
        self._identifier_bus = defaultdict()
        # Dict[Tuple[released stop id, route id, bus id], dynamic remaining hold time]
        self._identifier_time = defaultdict()
        self.holder_log = HolderLog(virtual_bus, has_schedule)
        self._has_schedule = has_schedule

    def add_bus(self, stop_id: str, bus: Bus, t: int) -> None:
        self._identifier_bus[(stop_id, bus.route_id, bus.bus_id)] = bus
        self._identifier_time[(stop_id, bus.route_id, bus.bus_id)] = None
        bus.set_status('holding')
        bus.update_location(t, 'holder', stop_id, stop_id, 0, 'holding')

    def set_hold_action(self, stop_bus_hold_action: Dict[Tuple[str, str, str], float]):
        for (stop_id, route_id, bus_id), hold_time in stop_bus_hold_action.items():
            assert self._identifier_time[(
                stop_id, route_id, bus_id)] is None, 'bus is already holding'
            self._identifier_time[(stop_id, route_id, bus_id)] = hold_time

    def operation(self, t: int) -> Dict[str, List[Bus]]:
        # store the buses that finished holding
        stop_held_buses = defaultdict(list)
        # store the buses with stop_id, route_id, bus_id, and remove them after the loop
        remove_buses = []
        for (stop_id, route_id, bus_id), hold_time in self._identifier_time.items():
            if hold_time is not None:
                # update holding time
                hold_time -= 1.0
                self._identifier_time[(stop_id, route_id, bus_id)] = hold_time
                held_bus = self._identifier_bus[(stop_id, route_id, bus_id)]

                # if holding is finished
                if hold_time <= 0:
                    # append the bus to the `stop_held_buses` and finally return to the `simulator`
                    stop_held_buses[stop_id].append(held_bus)

                    if self._has_schedule:
                        bus_id_seq = self.holder_log.route_stop_departure_bus_id_seq[
                            route_id][stop_id]
                        departure_idx_count = len(bus_id_seq)
                        epsilon_departure = held_bus.bus_log.record_when_departure(
                            stop_id, t, last_departure_idx_count=departure_idx_count)
                        self.holder_log.record_when_bus_departure(
                            stop_id, route_id, bus_id, t, epsilon_departure=epsilon_departure)
                    else:
                        epsilon_departure = held_bus.bus_log.record_when_departure(
                            stop_id, t)
                        assert epsilon_departure is None, 'epsilon_departure should be None'
                        self.holder_log.record_when_bus_departure(
                            stop_id, route_id, bus_id, t)

                    remove_buses.append((stop_id, route_id, bus_id))

                held_bus.update_location(
                    t, 'holder', stop_id, stop_id, 0, 'holding')

        for remove_bus in remove_buses:
            self._identifier_bus.pop(remove_bus)
            self._identifier_time.pop(remove_bus)

        return stop_held_buses

    def take_snapshot(self) -> HolderSnapshot:
        unheld_buses = self._find_unheld_buses()
        if self._has_schedule:
            holder_snapshot = HolderSnapshot(unheld_buses, self.holder_log.route_stop_departure_time_seq,
                                             self.holder_log.route_stop_departure_bus_id_seq, self.holder_log.route_stop_bus_epsilon_departure)
        else:
            holder_snapshot = HolderSnapshot(unheld_buses, self.holder_log.route_stop_departure_time_seq,
                                             self.holder_log.route_stop_departure_bus_id_seq)
        return holder_snapshot

    @property
    def stop_identifier_bus(self) -> Dict[Tuple[str, str, str], Bus]:
        return self._identifier_bus

    @property
    def has_schedule(self) -> bool:
        return self._has_schedule

    def _find_unheld_buses(self) -> List[Tuple[str, str, str]]:
        unheld_buses = []
        for (stop_id, route_id, bus_id), bus in self._identifier_bus.items():
            if self._identifier_time[(stop_id, route_id, bus_id)] is None:
                unheld_buses.append((stop_id, route_id, bus_id))
        return unheld_buses
