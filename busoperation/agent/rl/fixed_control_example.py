from copy import deepcopy
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
import random
import numpy as np
import torch

from simulator.snapshot import Snapshot
from setup.blueprint import Blueprint
from simulator.virtual_bus import VirtualBus
from simulator.simulator import Simulator

from .rl_agent import RLAgent
from .net import Actor_Net, Critic_Net


class Attention_DDPG(RLAgent):
    def __init__(self, agent_config: Dict[str, Any], blueprint: Blueprint) -> None:
        super().__init__(agent_config, blueprint)

    def calculate_hold_time(self, snapshot: Snapshot) -> Dict[Tuple[str], float]:
        stop_bus_hold_time = {}
        for (stop_id, route_id, bus_id) in snapshot.holder_snapshot.action_buses:
            stop_bus_hold_time[(stop_id, route_id, bus_id)] = 10

        return stop_bus_hold_time

    def reset(self, episode: int) -> None:
        pass

    def save_net(self, path: str) -> None:
        pass
        # torch.save(self._actor_net.state_dict(), path)

    def load_net(self, path):
        pass
        # self._actor_net.load_state_dict(torch.load(path))
