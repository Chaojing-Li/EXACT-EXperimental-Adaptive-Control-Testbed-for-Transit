import numpy as np
from scipy.stats import norm
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod

from setup.config_dataclass import LinkGeometry, LinkDistribution

from .bus import Bus


class Link(ABC):
    def __init__(self, link_id: str, link_geometry: LinkGeometry) -> None:
        self._link_id = link_id
        self._head_node = link_geometry.head_node
        self._tail_node = link_geometry.tail_node
        # traveling distance from terminal for the head node
        # self._distance_from_terminal = link_geometry.distance_from_terminal
        self._length = link_geometry.length
        # buses running on this link
        self._buses: List[Bus] = []

        # buses' relative locations (to the head_node) on this link
        self._bus_link_loc: Dict[Tuple[str, str], float] = {}

    def __repr__(self) -> str:
        return f"Link {self._link_id} from {self._head_node} to {self._tail_node}"

    @property
    def buses(self) -> List[Bus]:
        return self._buses

    # @property
    # def tail_node(self) -> str:
    #     return self._tail_node

    # accept a bus entering this link
    @abstractmethod
    def enter_bus(self, bus: Bus, t: int) -> None:
        ...

    # move buses one step (delta t) forward
    @abstractmethod
    def forward(self, t: int) -> List[Bus]:
        ...


class DistributionLink(Link):
    def __init__(self, link_id: str, link_geometry: LinkGeometry, link_distribution: LinkDistribution) -> None:
        super().__init__(link_id, link_geometry)

        self._tt_mean = link_distribution.tt_mean
        self._tt_cv = link_distribution.tt_cv
        self._tt_type = link_distribution.tt_type
        if self._tt_type == "normal":
            mu, sigma = self._tt_mean, self._tt_mean * self._tt_cv
            self._tt_distribution = norm(mu, sigma)

    def enter_bus(self, bus: Bus, t: int) -> None:
        # generate link travel time
        sampled_tt = self._tt_distribution.rvs(size=1).item()
        sampled_tt = max(10, sampled_tt)
        bus.bus_log.record_when_enter_link(
            self._link_id, sampled_tt-self._tt_mean)

        # self._bus_speed[(bus.route_id, bus.bus_id)] = self._length / sampled_tt
        bus.speed = self._length / sampled_tt
        self._buses.append(bus)

        # bus relative location (to the head node) on this link
        self._bus_link_loc[(bus.route_id, bus.bus_id)] = 0.0

        bus.update_location(t, 'link', self._link_id,
                            self._head_node, 0, 'running_on_link')
        bus.set_status('running_on_link')

    def forward(self, t: int) -> List[Bus]:
        finished_buses = []
        for bus in self._buses:
            self._bus_link_loc[(bus.route_id, bus.bus_id)] += bus.speed * 1.0

            offset = self._bus_link_loc[(bus.route_id, bus.bus_id)]
            bus.update_location(t, 'link', self._link_id,
                                self._head_node, offset, 'running_on_link')
            bus.set_status('running_on_link')

            if self._bus_link_loc[(bus.route_id, bus.bus_id)] >= self._length:
                self._bus_link_loc.pop((bus.route_id, bus.bus_id))
                finished_buses.append(bus)
                self._buses.remove(bus)
        return finished_buses
