from typing import Dict, Any

from setup.blueprint import Blueprint

from .agent import Agent


class DoNothing(Agent):
    def __init__(self, agent_config: Dict[str, Any], blueprint: Blueprint) -> None:
        super().__init__(agent_config)
        self._blueprint = blueprint

    def calculate_hold_time(self, snapshot):
        stop_bus_hold_time = {}
        for (stop_id, route_id, bus_id) in snapshot.holder_snapshot.action_buses:
            stop_bus_hold_time[(stop_id, route_id, bus_id)] = 0
        return stop_bus_hold_time

    def reset(self, episode: int):
        pass
