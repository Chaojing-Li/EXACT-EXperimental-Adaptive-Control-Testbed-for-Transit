from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from agent.agent import Agent
from setup.config_dataclass import StopNodeGeometry, StopNodeOperation
from simulator.virtual_bus import VirtualBus

from .pax_queue import PaxQueue
from .bus import Bus
from .pax import Pax
from .snapshot import StopSnapshot
from .log import StopLog


class Stop(ABC):
    ''' A Mixin interface defining the common methods for all types of stops.

    Attributes:
        stop_log: a StopLog object for logging

    '''

    def __init__(self, stop_id: str,
                 node_geometry: StopNodeGeometry,
                 node_operation: StopNodeOperation,
                 virtual_bus: VirtualBus,
                 has_schedule: bool):
        self._stop_id: str = stop_id
        self._berth_num: int = node_geometry.berth_num
        self._has_schedule: bool = has_schedule
        self._queue_rule: str = node_operation.queue_rule

        # queueing area for buses waiting to enter the berth
        self._entry_queue: List[Bus] = []

        # buses in berth:
        # if the berth is empty, then the value is None
        # if the berth is occupied, then the value is the bus
        # the index larger, the berth closer to the downstream
        self._buses_in_berth: List[Optional[Bus]] = [None] * self._berth_num

        # store buses finished service and used for processing the leaving event
        self._leave_queue: List[Bus] = []

        self.stop_log: StopLog = StopLog(
            self._stop_id, virtual_bus, has_schedule)

    @property
    @abstractmethod
    def _pax_queue(self) -> PaxQueue:
        ''' A `_pax_queue` attribute must be created by the subclass.

        '''
        ...

    @abstractmethod
    def _enter_berth(self, t: int) -> None:
        ''' Accept bus(es) from the entry queue to the berth.

        '''
        ...

    @abstractmethod
    def _board(self, t: int) -> None:
        ...

    @abstractmethod
    def _alight(self, t: int) -> List[Pax]:
        ...

    @abstractmethod
    def _check_leave(self, t: int):
        ...

    @abstractmethod
    def _leave(self, t: int) -> List[Bus]:
        ...

    def get_total_buses(self) -> List[Bus]:
        ''' Return all the buses in the stop, including buses in berth and buses in queue.

        Returns:
            List[Bus]: all the buses in the stop
        '''
        buses_in_queue = [bus for bus in self._entry_queue]
        buses_in_berth = [
            bus for bus in self._buses_in_berth if bus is not None]
        return buses_in_queue + buses_in_berth

    def take_snapshot(self) -> StopSnapshot:
        total_pax_num = self._pax_queue.get_total_pax_num()
        if self._has_schedule:
            stop_snapshot = StopSnapshot(self._stop_id, total_pax_num,
                                         self.stop_log.route_arrival_time_seq, self.stop_log.route_arrival_bus_id_seq,
                                         self.stop_log.route_rtd_time_seq, self.stop_log.route_rtd_bus_id_seq,
                                         self.stop_log.route_bus_epsilon_arrival, self.stop_log.route_bus_epsilon_rtd)
        else:
            stop_snapshot = StopSnapshot(self._stop_id, total_pax_num,
                                         self.stop_log.route_arrival_time_seq, self.stop_log.route_arrival_bus_id_seq,
                                         self.stop_log.route_rtd_time_seq, self.stop_log.route_rtd_bus_id_seq)
        return stop_snapshot

    def pax_arrive(self, paxs: List[Pax]):
        ''' Accept passengers that arrive at this stop.

        Args:
            paxs (List[Pax]): a list of passengers arriving at this stop

        '''
        for pax in paxs:
            self._pax_queue.add_pax(pax)

    def enter_stop(self, bus: Bus, t: int):
        ''' Accept a bus entering this stop.

        The bus will first enter the entry queue and then enter the berth.

        Args:
            bus (Bus): a bus entering this stop
            t (int): current time

        '''
        if self._has_schedule:
            # get the arrival index count at the stop for this bus on its route
            # this count is without the current bus
            # it will be used to retrieve the scheduled arrival time of the bus for calculating the schedule deviation
            arrival_idx_count = len(
                self.stop_log.route_arrival_time_seq[bus.route_id])

            # record the arrival time and schedule deviation into the bus's running log
            epsilon_arrival = bus.bus_log.record_when_arrival(
                self._stop_id, t, last_arrival_idx_count=arrival_idx_count)

            # recrod the arrival time and schedule deviation into the stop's log
            self.stop_log.record_when_bus_arrival(
                bus.route_id, bus.bus_id, t, epsilon_arrival=epsilon_arrival)
        else:
            epsilon_arrival = bus.bus_log.record_when_arrival(self._stop_id, t)
            assert epsilon_arrival is None, 'epsilon_arrival should be None'
            self.stop_log.record_when_bus_arrival(bus.route_id, bus.bus_id, t)

        self._entry_queue.append(bus)
        bus.set_status('queueing_at_stop')

    def operation(self, t: int) -> Tuple[List[Bus], List[Pax]]:
        ''' The main operation of the stop.

        '''
        self._enter_berth(t)
        self._board(t)
        leaving_paxs = self._alight(t)
        self._check_leave(t)
        leaving_buses = self._leave(t)
        self._pax_queue.accumulate_out_vehicle_delay()
        return leaving_buses, leaving_paxs

    def _queue_rule_check_in(self) -> Optional[int]:
        ''' Check a whether a bus (at the head of the entry queue) can enter the berth based on the queue rule.

        Get the target berth for the bus in the entry queue.

        Returns:
            target_berth: the target berth index. If no berth is available, return None

        '''

        target_berth = None
        if self._queue_rule == 'FO':
            for b in range(len(self._buses_in_berth) - 1, -1, -1):
                if self._buses_in_berth[b] == None:
                    target_berth = b
                else:
                    break
            return target_berth

        elif self._queue_rule == 'FIFO':
            for b in range(len(self._buses_in_berth)):
                if self._buses_in_berth[b] == None:
                    target_berth = b
                else:
                    break
        return target_berth

    def _queue_rule_check_out(self, berth_idx: int) -> bool:
        ''' Check if a bus at the berth index can leave the stop based on the queue rule.

        Args:
            berth_idx (int): the index of the berth where the bus is.

        Returns:
            bool: True if the bus can leave the stop, False otherwise

        '''

        can_go = False
        if self._queue_rule == 'FO':
            can_go = True
        elif self._queue_rule == 'FIFO':
            if berth_idx == self._berth_num-1:
                can_go = True
            else:
                for b in range(berth_idx+1, self._berth_num):
                    if self._buses_in_berth[b] != None:
                        break
                    else:
                        if b == self._berth_num-1:  # all the most downstream berths are clear
                            can_go = True
        return can_go
