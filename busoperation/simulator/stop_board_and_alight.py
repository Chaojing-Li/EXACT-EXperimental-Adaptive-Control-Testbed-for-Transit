from typing import List, Optional

from setup.config_dataclass import StopNodeGeometry, StopNodeOperation
from simulator.virtual_bus import VirtualBus

from .pax_queue import Pax
from .stop_board import Stop_Board


class Stop_Board_Alight(Stop_Board):
    def __init__(self, stop_id: str, stop_node_geometry: StopNodeGeometry,
                 stop_node_operation: StopNodeOperation, virtual_bus: VirtualBus,
                 has_schedule: bool):
        super().__init__(stop_id, stop_node_geometry,
                         stop_node_operation, virtual_bus, has_schedule)

        self._stop_id = stop_id

    def _alight(self, t: int) -> List[Pax]:
        leaving_paxs: List[Pax] = []
        for bus_in_berth in self._buses_in_berth:
            if bus_in_berth is None:
                continue
            leaving_pax = bus_in_berth.alight(self._stop_id)
            if leaving_pax is not None:
                leaving_paxs.append(leaving_pax)
                leaving_pax.alight_time = t

            bus_in_berth.update_location(
                t, 'stop', self._stop_id, self._stop_id, 0, 'dwelling_at_stop')
        return leaving_paxs

    def _check_leave(self, t: int):
        for berth_idx, bus_in_berth in enumerate(self._buses_in_berth):
            if bus_in_berth is None:
                continue
            remaining_pax_num_at_stop = self._pax_queue.check_remaining_pax_num(
                bus_in_berth)
            remaining_pax_num_at_bus = bus_in_berth.check_remaining_pax_num(
                self._stop_id)

            if remaining_pax_num_at_stop == 0 and remaining_pax_num_at_bus == 0:
                if self._queue_rule_check_out(berth_idx):
                    self._buses_in_berth[berth_idx] = None
                    self._leave_queue.append(bus_in_berth)
