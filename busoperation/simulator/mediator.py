from typing import Dict, List

from setup.blueprint import Blueprint

from .bus import Bus
from .holder import Holder
from .link import Link
from .stop import Stop
from .terminal import Terminal


class Mediator:
    def __init__(self, blueprint: Blueprint, terminals: Dict[str, Terminal],
                 links: Dict[str, Link], stops: Dict[str, Stop], holder: Holder) -> None:
        self._blueprint = blueprint
        self._terminals = terminals
        self._links = links
        self._stops = stops
        self._holder = holder

    def transfer(self, buses: List[Bus], spot_type: str, spot_id: str, t: int):
        for bus in buses:
            if spot_type == 'terminal':
                next_link_id = self._blueprint.get_next_link_id(
                    bus.route_id, spot_id)
                self._links[next_link_id].enter_bus(bus, t)

                if self._holder.has_schedule:
                    self._holder.holder_log.record_when_bus_departure(
                        spot_id, bus.route_id, bus.bus_id, t, epsilon_departure=0)
                else:
                    self._holder.holder_log.record_when_bus_departure(
                        spot_id, bus.route_id, bus.bus_id, t)

            elif spot_type == 'link':
                next_node_id, is_ending_terminal = self._blueprint.get_next_node_id(
                    bus.route_id, spot_id)
                if is_ending_terminal:
                    self._terminals[next_node_id].recycle(bus)
                    bus.bus_log.record_when_finish(t)
                else:
                    self._stops[next_node_id].enter_stop(bus, t)

            elif spot_type == 'stop':
                # if the bus need to be held in this stop, enter the holder
                # otherwise, directly enter the next link
                if spot_id in bus.hold_stops:
                    self._holder.add_bus(spot_id, bus, t)
                else:
                    next_link_id = self._blueprint.get_next_link_id(
                        bus.route_id, spot_id)
                    self._links[next_link_id].enter_bus(bus, t)

            elif spot_type == 'holder':
                next_link_id = self._blueprint.get_next_link_id(
                    bus.route_id, spot_id)
                self._links[next_link_id].enter_bus(bus, t)
