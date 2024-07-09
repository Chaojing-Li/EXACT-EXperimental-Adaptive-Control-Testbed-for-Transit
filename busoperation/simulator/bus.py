from typing import List, Literal, Dict, Optional

from simulator.virtual_bus import VirtualBus
from setup.route import Route_Details

from .log import BusRunningLog
from .pax import Pax
from .snapshot import BusSnapshot
from .trajectory import TrajectoryPoint


class Bus:
    ''' Represent a bus in the simulation.

    Attributes:
        bus_log: BusRunningLog
        speed: traversing speed on link, in meters/sec
        loc_relative_to_terminal: location relative to the terminal, in meters
        trajectory: time-point trajectory for plotting
        route_id: route id
        bus_id: bus id
        board_status: boarding status, either 'boarding' or 'idle'

    Methods:
        set_status(self, status: Literal['running_on_link', 'queueing_at_stop', 'dwelling_at_stop', 'holding', 'finished'])
        accumate_board_fraction(self) -> None
        board(self, pax: Pax) -> None
        alight(self, stop_id: str) -> None
        check_remaining_pax_num(self, stop_id: str) -> int
        update_location(self, t: int, spot_type: str, spot_id: str, node_id: str, offset: float) -> None
        take_snapshot(self) -> BusSnapshot

    '''

    def __init__(self, bus_id: str,
                 route: Route_Details,
                 node_distance: Dict[str, float],
                 virtual_bus: VirtualBus,
                 is_need_to_hold: bool
                 ) -> None:

        self._bus_id: str = bus_id
        self._route_id: str = route.route_id
        self._capacity: int = route.bus_capacity
        # the distance of each node from the terminal
        self._node_distance: Dict[str, float] = node_distance

        self._status: Literal['running_on_link', 'queueing_at_stop',
                              'dwelling_at_stop', 'holding', 'finished'] = 'running_on_link'
        self._board_status: Literal['boarding', 'idle'] = 'idle'
        self._alight_status: Literal['alighting', 'idle'] = 'idle'

        # onboard pax list
        self._paxs: List[Pax] = []
        # boarding fraction indicate how much of a single boarding action has been completed
        self._board_fraction: float = 0.0
        # the boarding rate determined by the boarding pax
        self._pax_board_rate: Optional[float] = None

        # alighting fraction indicate how much of a single alighting action has been completed
        self._alight_fraction: float = 0.0
        # the alighting rate determined by the alighting pax
        self._pax_alight_rate: Optional[float] = None

        self._trajectory: Dict[int, TrajectoryPoint] = {}
        # indicate whether the bus is need to hold at stops
        self._is_need_to_hold: bool = is_need_to_hold

        # indicate in which stops the bus should be held
        self._hold_stops: List[str] = route.hold_stops

        virtual_bus_stop_arrival_time = virtual_bus.route_stop_arrival_time[self._route_id]
        virtual_bus_stop_rtd_time = virtual_bus.route_stop_rtd_time[self._route_id]
        virtual_bus_stop_departure_time = virtual_bus.route_stop_departure_time[self._route_id]

        self.bus_log: BusRunningLog = BusRunningLog(
            route.schedule_headway, virtual_bus_stop_arrival_time, virtual_bus_stop_rtd_time, virtual_bus_stop_departure_time)
        self.speed: float = 0.0
        self.loc_relative_to_terminal: float = 0.0

    def __repr__(self) -> str:
        return f'Bus {self._bus_id} on route {self._route_id} with pax_num {len(self._paxs)}'

    @property
    def trajectory(self) -> Dict[int, TrajectoryPoint]:
        ''' Trajectory point at each time step.

        Key is the time step, value is the TrajectoryPoint object.

        '''
        return self._trajectory

    @property
    def route_id(self) -> str:
        return self._route_id

    @property
    def bus_id(self) -> str:
        return self._bus_id

    @property
    def board_status(self) -> Literal['boarding', 'idle']:
        return self._board_status

    @property
    def is_need_to_hold(self) -> bool:
        return self._is_need_to_hold

    @property
    def status(self) -> Literal['running_on_link', 'queueing_at_stop', 'dwelling_at_stop', 'holding', 'finished']:
        return self._status

    @property
    def hold_stops(self) -> List[str]:
        return self._hold_stops

    def set_status(self, status: Literal['running_on_link', 'queueing_at_stop', 'dwelling_at_stop', 'holding', 'finished']) -> None:
        self._status = status

    def accumate_board_fraction(self) -> None:
        ''' Accumulate the board fraction for bus that is boarding pax.

        The bus will be first set to 'boarding' status from 'idle' status, and then the board fraction will be accumulated.
        After the board fraction reaches 1, the bus will be set to 'idle' status again.
        The process will be repeated for the next boarding pax.

        '''
        assert self._board_status == 'boarding', 'bus is not boarding, cannot accumate board fraction'
        assert self._pax_board_rate is not None, 'pax board rate is None'
        self._board_fraction += self._pax_board_rate
        if self._board_fraction >= 1:
            self._board_fraction -= 1
            self._board_status = 'idle'
            self._pax_board_rate = None

    def board(self, pax: Pax, t: int) -> None:
        ''' Board pax onto the bus.

        The pax is first added to the onboard pax list, 
        and then the bus is set to 'boarding' status and boarding fraction will be accumulated.

        Args:
            pax: the pax to board onto the bus

        '''
        self._board_status = 'boarding'
        self._pax_board_rate = pax.board_rate
        pax.board_time = t
        self._paxs.append(pax)

    def alight(self, stop_id: str) -> Optional[Pax]:
        ''' Alight pax from the bus.

        Args:
            stop_id: the id of the stop where pax alight from the bus

        '''

        leaving_pax = None
        if self._alight_status == 'alighting':
            assert self._pax_alight_rate is not None, 'pax alight rate is None'
            self._alight_fraction += self._pax_alight_rate
            if self._alight_fraction >= 1:
                self._alight_fraction -= 1
                self._alight_status = 'idle'
                self._pax_alight_rate = None
        elif self._alight_status == 'idle':
            for pax in self._paxs:
                if pax.destination == stop_id:
                    # first remove the pax and then set the alight status to 'alighting' to count the alight fraction
                    self._paxs.remove(pax)
                    leaving_pax = pax
                    self._pax_alight_rate = pax.alight_rate
                    self._alight_status = 'alighting'
                    break
            else:
                pass
        return leaving_pax

    def accumulate_in_vehicle_delay(self) -> None:
        ''' Accumulate in-vehicle delay for passengers on the bus.

        '''
        for pax in self._paxs:
            pax.accumulate_in_vehicle_delay()

    def check_remaining_pax_num(self, stop_id: str) -> int:
        ''' Check the remaining pax number in the bus that are going to alight at the stop.

        Args:
            stop_id: the id of the stop where pax alight from the bus

        Returns:
            int: the number of pax that are going to alight at the stop

        '''
        return len([pax for pax in self._paxs if pax.destination == stop_id])

    def update_location(self, t: int,
                        spot_type: str,
                        spot_id: str,
                        node_id: str,
                        offset: float,
                        status: Literal['running_on_link', 'queueing_at_stop',
                                        'dwelling_at_stop', 'holding', 'finished']
                        ) -> None:
        ''' Update bus's relative location to the terminal.

        One purpose is to record trajectory for plotting.
        Another purpose is to provide a partial state of the current bus, for the control agent to make decisions.

        Args:
            t: current time
            spot_type: 'link', 'node' or 'holder'
            spot_id: the id of the spot
            offset: for the spot_type of 'link', the offset from the head node
                    for the spot_type of 'stop', offset=0

        '''
        self.loc_relative_to_terminal = self._node_distance[node_id] + offset
        self._trajectory[t] = TrajectoryPoint(
            spot_type, spot_id, self.loc_relative_to_terminal, status)

    def take_snapshot(self) -> BusSnapshot:
        '''Take a snapshot of the bus at the current time step.

        Pax_num, loc_relative_to_terminal and status are instantaneous values.
        Dwell times, link tt deviation, epsilon arrival and epsilon rtd contains all the historical values by `RunningLog`

        Returns:
            BusSnapshot: a snapshot of the bus at the current time step

        '''

        pax_num = len(self._paxs)
        bus_snapshot = BusSnapshot(self._bus_id, self._route_id, self._is_need_to_hold, pax_num, self.loc_relative_to_terminal,
                                   self._status, dict(
                                       self.bus_log.stop_dwell_time), self.bus_log.link_tt_deviation,
                                   self.bus_log.stop_epsilon_arrival, self.bus_log.stop_epsilon_rtd,
                                   self.bus_log.stop_epsilon_departure, self.bus_log.visited_stops)
        return bus_snapshot
