from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Literal
import numpy as np
from scipy.stats import norm
from collections import defaultdict

from setup.route import Route_Schema
from setup.config_dataclass import PaxOperation
from simulator.virtual_bus import VirtualBus


@dataclass
class Pax:
    pax_id: str
    origin: str
    destination: str
    # the list of routes that can take pax from origin to destination
    routes: List[str]
    arrival_time: int
    board_rate: float

    # the rate that pax alights from the vehicle, pax/second
    alight_rate: Optional[float]
    # the time that pax boards the vehicle
    board_time: Optional[int] = None
    # the time that pax alights from the vehicle
    alight_time: Optional[int] = None

    # used to accumualte the out vehicle delay time each second
    out_vehicle_delay: int = 0
    # used to accumualte the in vehicle delay time each second
    in_vehicle_delay: int = 0

    def __repr__(self) -> str:
        return f'Pax {self.pax_id} from {self.origin} to {self.destination} on {self.routes}, arrived at {self.arrival_time}'

    def accumulate_out_vehicle_delay(self):
        self.out_vehicle_delay += 1

    def accumulate_in_vehicle_delay(self):
        self.in_vehicle_delay += 1


class PaxGenerator:

    def __init__(self, route_scheme: Route_Schema, pax_operation: PaxOperation, virtual_bus: VirtualBus) -> None:
        self._route_od_table = route_scheme.route_OD_rate_table
        self._route_stop_pax_arrival_start_time = virtual_bus.route_stop_pax_arrival_start_time

        self._pax_arrival_type = pax_operation.pax_arrival_type
        self._pax_board_time_mean = pax_operation.pax_board_time_mean
        self._pax_board_time_std = pax_operation.pax_board_time_std
        self._pax_board_time_type = pax_operation.pax_board_time_type

        self._pax_alight_time_mean = pax_operation.pax_alight_time_mean
        self._pax_alight_time_std = pax_operation.pax_alight_time_std
        self._pax_alight_time_type = pax_operation.pax_alight_time_type

        if self._pax_arrival_type == 'deterministic':
            self._route_od_arrival_marker: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(
                lambda: defaultdict(float))

            self._route_origin_arrival_marker: Dict[str, Dict[str, float]] = defaultdict(
                lambda: defaultdict(float))

        if self._pax_board_time_type == 'normal':
            mu, sigma = self._pax_board_time_mean, self._pax_board_time_std
            self._board_time_distribution = norm(mu, sigma)

        if self._pax_alight_time_type == 'normal':
            mu, sigma = self._pax_alight_time_mean, self._pax_alight_time_std
            self._alight_time_distribution = norm(mu, sigma)

        self._pax_count = 0

    def generate(self, t: int) -> Dict[str, List[Pax]]:
        # TODO search common routes between origin and destination
        stop_paxs = defaultdict(list)
        for route_id, od_table in self._route_od_table.items():
            for origin_stop_id, dest_stop_od in od_table.items():
                if t <= self._route_stop_pax_arrival_start_time[route_id][origin_stop_id]:
                    continue

                if self._pax_arrival_type == 'deterministic':
                    pax_num, dest_stop = self._deterministic_generation(
                        route_id, origin_stop_id, dest_stop_od)
                    assert pax_num <= 1, 'Only one pax can be generated at a time for deterministic arrival'

                    if pax_num == 1:
                        assert dest_stop is not None
                        board_rate = self._get_board_rate()
                        if self._pax_alight_time_type is not None:
                            alight_rate = self._get_alight_rate()
                        else:
                            alight_rate = None
                        pax = Pax(str(self._pax_count), origin_stop_id,
                                  dest_stop, [route_id], t, board_rate, alight_rate)
                        stop_paxs[origin_stop_id].append(pax)
                        self._pax_count += 1

                elif self._pax_arrival_type == 'poisson':
                    for dest_stop_id, rate in dest_stop_od.items():
                        common_routes = [route_id]
                        pax_num = 0
                        pax_num = self._get_poission_pax_num(rate)
                        for pax in range(pax_num):
                            board_rate = self._get_board_rate()
                            if self._pax_alight_time_type is not None:
                                alight_rate = self._get_alight_rate()
                            else:
                                alight_rate = None
                            pax = Pax(str(self._pax_count), origin_stop_id,
                                      dest_stop_id, common_routes, t, board_rate, alight_rate)
                            stop_paxs[origin_stop_id].append(pax)
                            self._pax_count += 1
        return dict(stop_paxs)

    def _deterministic_generation(self, route_id: str,
                                  origin_stop_id: str,
                                  dest_stop_rate: Dict[str, float]
                                  ) -> Tuple[Literal[0, 1], Optional[str]]:
        current_rate = self._route_origin_arrival_marker[route_id][origin_stop_id]
        total_arrival_rate = sum(dest_stop_rate.values())
        new_rate = current_rate + total_arrival_rate
        if new_rate >= 1:
            new_rate -= 1
            # Extract the keys (destinations) and values (rates) from the dictionary
            destinations = list(dest_stop_rate.keys())
            rates = list(dest_stop_rate.values())
            # Normalize the rates to obtain valid probabilities
            total_rate = sum(rates)
            probabilities = [rate / total_rate for rate in rates]
            # Sample a destination based on the probabilities
            sampled_destination = np.random.choice(
                destinations, p=probabilities)
            self._route_origin_arrival_marker[route_id][origin_stop_id] = new_rate
            return 1, sampled_destination
        else:
            self._route_origin_arrival_marker[route_id][origin_stop_id] = new_rate
            return 0, None

    def _get_poission_pax_num(self, rate: float) -> int:
        return np.random.poisson(rate)

    def _get_board_rate(self):
        if self._pax_board_time_type == 'deterministic':
            return 1 / self._pax_board_time_mean
        else:
            sampled_time = self._board_time_distribution.rvs(size=1).item()
            sampled_time = max(0.1, sampled_time)
            sampled_time = min(10, sampled_time)
            return 1/sampled_time

    def _get_alight_rate(self):
        assert self._pax_alight_time_mean is not None
        if self._pax_alight_time_type == 'deterministic':
            return 1 / self._pax_alight_time_mean
        else:
            sampled_time = self._alight_time_distribution.rvs(size=1).item()
            sampled_time = max(0.1, sampled_time)
            sampled_time = min(10, sampled_time)
            return 1/sampled_time

    # def _get_deterministic_pax_num(self, route_id: str, origin_stop_id: str, dest_stop_id: str, rate: float) -> int:
    #     current_rate = self._route_od_arrival_marker[route_id][(
    #         origin_stop_id, dest_stop_id)]
    #     new_rate = current_rate + rate
    #     if new_rate >= 1:
    #         new_rate -= 1
    #         self._route_od_arrival_marker[route_id][(
    #             origin_stop_id, dest_stop_id)] = new_rate
    #         return 1
    #     else:
    #         self._route_od_arrival_marker[route_id][(
    #             origin_stop_id, dest_stop_id)] = new_rate
    #         return 0
