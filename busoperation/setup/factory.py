from typing import Protocol, Dict, Tuple

from agent.agent import Agent
from simulator.terminal import Terminal
from simulator.link import Link
from simulator.stop import Stop
from simulator.pax import PaxGenerator
from simulator.virtual_bus import VirtualBus

from .blueprint import Blueprint


class ComponentFactory(Protocol):
    ''' Protocol for creating components for the simulation environment

    '''

    def create_virtual_bus(self, blueprint: Blueprint, agent: Agent) -> VirtualBus:
        ''' An abstract method to create a virtual bus with the given blueprint and agent.

        '''
        ...

    def create_terminals(self, blueprint: Blueprint, virtual_bus: VirtualBus, hold_period: Tuple) -> Dict[str, Terminal]:
        ''' An abstract method to create terminals with the given blueprint, virtual bus and hold period.

        The hold period is a tuple of two integers, representing the start and end time of the hold period. 
        When a bus dispatched from the terminal within the hold period, it will be held at designated stop for a certain time.

        '''
        ...

    def create_stops(self, blueprint: Blueprint, virtual_bus: VirtualBus, has_schedule: bool) -> Dict[str, Stop]:
        ''' An abstract method to create stops with the given blueprint and virtual bus.

        If `has_schedule` is True, the stop's `StopLog` will maintain a schedule. 
        The schedule is calculated by shifting the virtual bus's schedule by n*dispatching headways, where n is the bus's index dispatched from the terminal.

        '''
        ...

    def create_links(self, blueprint: Blueprint) -> Dict[str, Link]:
        ''' An abstract method to create links with the given blueprint.

        '''
        ...

    def create_pax_generator(self, blueprint: Blueprint, virtual_bus: VirtualBus) -> PaxGenerator:
        ''' An abstract method to create a pax generator with the given pax operation and virtual bus.

        The virtual bus is used to specify the initial condition of the dynamics, i.e., the pax arrival process starts after when the virtual bus visits the first stop.

        '''
        ...
