from typing import List, Dict, Tuple, Literal
from collections import defaultdict
from copy import deepcopy

from .pax import Pax
from .bus import Bus


class PaxQueue:
    ''' A queue that holds paxs that are waiting for buses at a stop.

    Methods:

    '''

    def __init__(self, stop_id: str, board_truncation: Literal['arrival', 'rtd']):
        # Key is a group (tuple) of routes, value is a list of paxs that can be served by any route in the tuple
        # if the tuple contains only one route, then the paxs are exclusive
        # The name `group`` is used to indicate the paxs that share totally the same routes
        self._route_group_paxs: Dict[Tuple[str, ...],
                                     List[Pax]] = defaultdict(list)

        # the stop id that the queue belongs to
        self._stop_id: str = stop_id

        # the filter type to filter paxs that can be served by the bus
        self._board_truncation: Literal['arrival', 'rtd'] = board_truncation

    def add_pax(self, pax: Pax):
        ''' Add a pax to the queue.

        Args:
            pax: the pax to be added to the queue

        '''
        routes = tuple(pax.routes)
        self._route_group_paxs[routes].append(pax)

    def board(self, bus: Bus, t: int):
        ''' Board paxs in this queue to a bus

        Args:
            bus: the bus to board paxs
        '''
        # the buse has two `board_status`: boarding and idle
        # if the bus is boarding, then it is in the middle of boarding
        if bus.board_status == 'boarding':
            # board a fraction of a pax in the queue
            bus.accumate_board_fraction()
            return
        # if the bus is idle, then it is ready to board
        elif bus.board_status == 'idle':
            # 0. find pax groups that the bus can serve
            served_groups = self._get_served_groups(bus.route_id)
            # no group can be served
            if len(served_groups) == 0:
                return

            # 1. serve the exlusive groups first
            exclusive_group = served_groups[0]
            assert len(exclusive_group) == 1, 'Only one exclusive group for now'
            paxs = self._route_group_paxs[exclusive_group]
            board_paxs = []
            # if `self._boarding_runcation` is 'arrival', then only board paxs that arrive before the bus arrives
            # i.e., filter out paxs that arrive after the bus arrives
            if self._board_truncation == 'arrival':
                board_paxs = self._filter_pax_arrival_after_bus_arrival(
                    bus, paxs)
            else:
                board_paxs = paxs

            if len(board_paxs) == 0:
                return
            # put the pax in the head of the queue on board, but the boarding process is not finished
            head_pax = board_paxs[0]
            # the bus's boarding status will be set to 'boarding' in this bus's board method
            bus.board(head_pax, t)
            paxs.remove(head_pax)
            bus.accumate_board_fraction()

        # TODO 2. serve common-line groups, for now, there is only one exclusive group

    def accumulate_out_vehicle_delay(self):
        for group, paxs in self._route_group_paxs.items():
            for pax in paxs:
                pax.accumulate_out_vehicle_delay()

    def get_total_pax_num(self) -> int:
        ''' Get the total number of paxs for all the routes

        Returns:
            The total number of paxs for all the routes
        '''
        total_pax_sum = sum([len(self._route_group_paxs[group])
                             for group in self._route_group_paxs.keys()])
        return total_pax_sum

    def check_remaining_pax_num(self, bus: Bus):
        '''Check if there are remaining paxs that can be served by the bus.

        Args:
            bus: the bus to be checked

        Returns:
            The number of remaining paxs that can be served by the bus
        '''
        served_groups = self._get_served_groups(bus.route_id)
        remaining_pax_num = 0
        for group in served_groups:
            paxs = self._route_group_paxs[group]
            board_paxs = []
            if self._board_truncation == 'arrival':
                board_paxs = self._filter_pax_arrival_after_bus_arrival(
                    bus, paxs)
            else:
                board_paxs = paxs
            remaining_pax_num += len(board_paxs)
        return remaining_pax_num

    def _get_served_groups(self, bus_route_id: str) -> List[Tuple[str, ...]]:
        '''Get all the served groups of paxs for a bus given its route id

        Args:
            bus_route_id: the route id of the bus

        Returns:
            A list of tuple, where each tuple is a group of routes that the bus can serve
        '''
        served_groups = []
        for route_group in self._route_group_paxs.keys():
            if bus_route_id in route_group:
                served_groups.append(route_group)
        return served_groups

    def _filter_pax_arrival_after_bus_arrival(self, bus: Bus, paxs: List[Pax]) -> List[Pax]:
        filtered_paxs = [pax for pax in paxs if pax.arrival_time <
                         bus.bus_log.stop_arrival_time[self._stop_id]]
        return filtered_paxs
