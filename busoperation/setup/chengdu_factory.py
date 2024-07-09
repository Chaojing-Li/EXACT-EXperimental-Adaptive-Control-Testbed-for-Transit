from typing import Dict, Literal, Tuple

from simulator.stop import Stop
from simulator.stop_board import Stop_Board
from simulator.stop_board_and_alight import Stop_Board_Alight
from simulator.link import Link, DistributionLink
from simulator.terminal import Terminal
from simulator.pax import PaxGenerator
from simulator.virtual_bus import VirtualBus
from agent.agent import Agent

from .blueprint import Blueprint
from .config_dataclass import StopNodeOperation, TerminalNodeOperation, PaxOperation

is_alight: bool = True

pax_arrival_type: Literal['deterministic', 'poisson'] = 'poisson'
pax_board_time_mean = 4.0
pax_board_time_std = 1.0
pax_board_time_type: Literal['deterministic', 'normal'] = 'normal'

# if alight is True, then pax_alight_time_mean and pax_alight_time_std are used
# otherwise, they are not used
pax_alight_time_mean = 3.0
pax_alight_time_std = 1.0
pax_alight_time_type: Literal['deterministic', 'normal'] = 'normal'

board_truncation: Literal['arrival', 'rtd'] = 'rtd'


class CD_Route3_Components_Factory:
    ''' Create components for CDRoute3 environment, adhering to the `ComponentFactory` Protocol

    '''

    def __init__(self, blueprint: Blueprint) -> None:
        self._stop_node_operation: Dict[str, StopNodeOperation] = {}
        for stop_id in blueprint.network.stop_node_geometry_info:
            node_operation = StopNodeOperation(
                is_alight=is_alight, queue_rule='FO', board_truncation=board_truncation)
            self._stop_node_operation[stop_id] = node_operation

        self._pax_operation = PaxOperation(pax_arrival_type=pax_arrival_type, pax_board_time_mean=pax_board_time_mean,
                                           pax_board_time_std=pax_board_time_std, pax_board_time_type=pax_board_time_type,
                                           pax_alight_time_mean=pax_alight_time_mean, pax_alight_time_std=pax_alight_time_std,
                                           pax_alight_time_type=pax_alight_time_type)

    def create_virtual_bus(self, blueprint: Blueprint, agent: Agent) -> VirtualBus:
        ''' Create a virtual bus with perfect schedule without repeated simulation

        '''
        # the agent has not created a virtual bus, created virtual bus with perfect schedule without repeated simulation
        virtual_bus = VirtualBus(blueprint)
        slack = 0
        virtual_bus.initialize_with_perfect_schedule(
            blueprint.route_stop_arrival_rate, slack)
        print('the agent has not created a virtual bus, created virtual bus with perfect schedule without repeated simulation')

        return virtual_bus

    def create_pax_generator(self, blueprint: Blueprint, virtual_bus: VirtualBus) -> PaxGenerator:
        pax_generator = PaxGenerator(
            blueprint.route_schema, self._pax_operation, virtual_bus)
        return pax_generator

    def create_terminals(self, blueprint: Blueprint, virtual_bus: VirtualBus, hold_period: Tuple) -> Dict[str, Terminal]:
        terminals = {}
        for terminal_id, routes in blueprint.route_schema.terminal_to_routes_info.items():
            terminal_node_geometry = blueprint.network.terminal_node_geometry_info[terminal_id]
            terminal_node_operation = TerminalNodeOperation()
            terminal = Terminal(terminal_id, terminal_node_geometry,
                                terminal_node_operation, routes, blueprint, virtual_bus, hold_period)
            terminals[terminal_id] = terminal
        return terminals

    def create_links(self, blueprint: Blueprint) -> Dict[str, Link]:
        links = {}
        for link_id, link_geometry in blueprint.network.link_geometry_info.items():
            link_distribution = blueprint.network.link_distribution[link_id]
            link = DistributionLink(link_id, link_geometry, link_distribution)
            links[link_id] = link
        return links

    def create_stops(self, blueprint: Blueprint, virtual_bus: VirtualBus, has_schedule: bool) -> Dict[str, Stop]:
        stops = {}
        for stop_id, stop_node_geometry in blueprint.network.stop_node_geometry_info.items():
            node_operation = self._stop_node_operation[stop_id]
            if is_alight:
                stop = Stop_Board_Alight(stop_id, stop_node_geometry,
                                         node_operation, virtual_bus, has_schedule)
            else:
                stop = Stop_Board(stop_id, stop_node_geometry,
                                  node_operation, virtual_bus, has_schedule)

            stops[stop_id] = stop
        return stops
