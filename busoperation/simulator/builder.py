from typing import Dict, Tuple

from agent.agent import Agent
from setup.homo_one_route_factory import Homo_One_Route_Components_Factory
from setup.chengdu_factory import CD_Route3_Components_Factory
from setup.guangzhou_brt_factory import GBRT_Components_Factory
from setup.factory import ComponentFactory
from setup.blueprint import Blueprint
from simulator.virtual_bus import VirtualBus

from .terminal import Terminal
from .link import Link
from .stop import Stop
from .pax import PaxGenerator


class Builder:
    ''' Create simulation components by factory.

    These components are customized by different env setting, by implementing `ComponentFactory` protocol

    Methods:
        create_terminals(self, agent: Agent) -> Dict[str, Terminal]:
        create_stops(self, agent: Agent) -> Dict[str, Stop]:
        create_links(self) -> Dict[str, Link]:
        create_pax_generator(self, agent: Agent) -> PaxGenerator:
        create_virtual_bus(self, agent: Agent) -> VirtualBus

    '''

    def __init__(self, blueprint: Blueprint, agent: Agent) -> None:
        self._blueprint: Blueprint = blueprint
        self._agent: Agent = agent
        if blueprint.env_name == 'homogeneous_one_route':
            self._component_factory: ComponentFactory = Homo_One_Route_Components_Factory(
                blueprint)
        elif blueprint.env_name == 'cd_route_3':
            self._component_factory: ComponentFactory = CD_Route3_Components_Factory(
                blueprint)
        elif blueprint.env_name == 'gbrt':
            self._component_factory: ComponentFactory = GBRT_Components_Factory(
                blueprint)

    def create_virtual_bus(self) -> VirtualBus:
        virtual_bus = self._component_factory.create_virtual_bus(
            self._blueprint, self._agent)
        return virtual_bus

    def create_terminals(self, virtual_bus: VirtualBus, hold_period: Tuple) -> Dict[str, Terminal]:
        terminals = self._component_factory.create_terminals(
            self._blueprint, virtual_bus, hold_period)
        return terminals

    def create_stops(self, virtual_bus: VirtualBus, has_schedule: bool) -> Dict[str, Stop]:
        stops = self._component_factory.create_stops(
            self._blueprint, virtual_bus, has_schedule)
        return stops

    def create_links(self) -> Dict[str, Link]:
        links = self._component_factory.create_links(self._blueprint)
        return links

    def create_pax_generator(self, virtual_bus: VirtualBus) -> PaxGenerator:
        pax_generator = self._component_factory.create_pax_generator(
            self._blueprint, virtual_bus)
        return pax_generator
