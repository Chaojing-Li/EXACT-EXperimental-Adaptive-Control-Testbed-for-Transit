from typing import List, Optional

from agent.agent import Agent
from setup.config_dataclass import StopNodeGeometry, StopNodeOperation
from simulator.virtual_bus import VirtualBus

from .bus import Bus
from .pax import Pax
from .pax_queue import PaxQueue
from .stop import Stop


class Stop_Board(Stop):
    def __init__(self, stop_id: str,
                 stop_node_geometry: StopNodeGeometry,
                 stop_node_operation: StopNodeOperation,
                 virtual_bus: VirtualBus,
                 has_schedule: bool):
        super().__init__(stop_id, stop_node_geometry,
                         stop_node_operation, virtual_bus, has_schedule)
        self._queue_rule = stop_node_operation.queue_rule
        # self._is_alight = stop_node_operation.is_alight
        self._board_truncation = stop_node_operation.board_truncation
        self._board_pax_queue = PaxQueue(stop_id, self._board_truncation)

    @property
    def _pax_queue(self) -> PaxQueue:
        ''' Implement the abstract property method from the parent class

        '''
        return self._board_pax_queue

    def _enter_berth(self, t: int) -> None:
        if len(self._entry_queue) == 0:
            return

        head_bus = self._entry_queue[0]
        target_berth = self._queue_rule_check_in()
        if target_berth is not None:
            self._buses_in_berth[target_berth] = head_bus
            head_bus.set_status('dwelling_at_stop')
            self._entry_queue.pop(0)

        for bus in self._entry_queue:
            bus.bus_log.record_when_queue(self._stop_id)
            bus.update_location(t, 'stop', self._stop_id,
                                self._stop_id, 0, 'queueing_at_stop')

    def _board(self, t: int) -> None:
        for bus_in_berth in self._buses_in_berth:
            if bus_in_berth is None:
                continue
            self._pax_queue.board(bus_in_berth, t)
            bus_in_berth.bus_log.record_when_dwell(self._stop_id)

            bus_in_berth.update_location(
                t, 'stop', self._stop_id, self._stop_id, 0, 'dwelling_at_stop')

    def _alight(self, t: int) -> List[Pax]:
        ''' No need to implement this method for boarding stop
        '''
        return []

    def _check_leave(self, t: int):
        ''' Check if the bus can leave the stop.

        '''

        for berth_idx, bus_in_berth in enumerate(self._buses_in_berth):
            if bus_in_berth is None:
                continue
            remaining_pax_num = self._pax_queue.check_remaining_pax_num(
                bus_in_berth)

            if remaining_pax_num == 0:
                can_go = self._queue_rule_check_out(berth_idx)
                if can_go:
                    self._buses_in_berth[berth_idx] = None
                    self._leave_queue.append(bus_in_berth)

    def _leave(self, t: int) -> List[Bus]:
        leaving_buses: List[Bus] = []
        for bus in self._leave_queue:
            if self._has_schedule:
                rtd_idx_count = len(
                    self.stop_log.route_rtd_bus_id_seq[bus.route_id])
                epsilon_rtd = bus.bus_log.record_when_rtd(
                    self._stop_id, t, last_rtd_idx_count=rtd_idx_count)
                self.stop_log.record_when_bus_rtd(
                    bus.route_id, bus.bus_id, t, epsilon_rtd=epsilon_rtd)
            else:
                epsilon_rtd = bus.bus_log.record_when_rtd(self._stop_id, t)
                assert epsilon_rtd is None, 'epsilon_rtd should be None'
                self.stop_log.record_when_bus_rtd(bus.route_id, bus.bus_id, t)

            leaving_buses.append(bus)
            self._leave_queue.remove(bus)
        return leaving_buses
