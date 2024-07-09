from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List, Literal

from agent.rl.helper import find_for_and_backward_buses
from simulator.snapshot import Snapshot
from simulator.virtual_bus import VirtualBus


class Agent(ABC):
    """
    A Mixin interface declares operations common to all supported versions of holding agents.

    Attributes:
        virtual_bus: Virtual bus that defines the starting of pax arrival at each stop

    """

    _virtual_bus: VirtualBus

    def __init__(self, agent_config: Dict[str, Any]) -> None:
        self._agent_name = agent_config['agent_name']

    @property
    def virtual_bus(self):
        ''' The virtual bus is created by subclasses of Agent.

        The property does not exist if the subclass does not implement it.

        '''
        return self._virtual_bus

    @abstractmethod
    def reset(self, episode: int) -> None:
        ''' Reset the agent for the next episode
        '''
        ...

    @abstractmethod
    def calculate_hold_time(self, snapshot: Snapshot) -> Dict[Tuple[str, str, str], float]:
        ''' Given the snapshot of the current state, calculate the hold time of each bus at each stop.
            Note that the snapshot also contains historical information, e.g., the last bus's arrival time at each stop.
            See the `Snapshot` class for more details.

        Args:
            snapshot: Snapshot

        Returns:
            stop_bus_hold_action: a dict mapping the tuple (stop_id, route_id, bus_id) to a determined holding time

        '''
        ...

    def extract_local_info_from_snapshot(self,
                                         curr_bus_id: str,
                                         snapshot: Snapshot,
                                         infos: List[Literal['spacing']]):
        ''' Extract local information from the snapshot

        Args:
            curr_bus_id: the query bus id
            snapshot: Snapshot
            infos: the list of information to extract

        '''
        bus_id_loc: Dict[str, float] = {}
        for (route_id, bus_id), bus_snapshot in snapshot.bus_snapshots.items():
            bus_id_loc[bus_id] = bus_snapshot.loc_relative_to_terminal
        forward_bus_id, forward_spacing, backward_bus_id, backward_spacing = find_for_and_backward_buses(
            bus_id_loc, curr_bus_id)

        return forward_bus_id, forward_spacing, backward_bus_id, backward_spacing

    def extract_global_info_from_snapshot(self,
                                          snapshot: Snapshot,
                                          infos: List[Literal['loc']]):
        ''' Extract global information from the snapshot

        Args:
            snapshot: Snapshot
            infos: the list of information to extract

        '''
        bus_id_loc: Dict[str, float] = {}
        for (route_id, bus_id), bus_snapshot in snapshot.bus_snapshots.items():
            bus_id_loc[bus_id] = bus_snapshot.loc_relative_to_terminal
        spacings = list(bus_id_loc.values())
        spacings.sort()
        return spacings
