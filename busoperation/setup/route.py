from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from dataclasses import dataclass

INF = int(1e8)


@dataclass
class Route_Details:
    ''' A dataclass for holding a specific route's information.

    '''
    route_id: str
    terminal_id: str
    visit_seq_stops: List[str]
    end_terminal_id: str
    od_rate_table: Dict[str, Dict[str, float]]
    schedule_headway: float
    schedule_headway_std: float
    boarding_rate: Dict[str, float]
    bus_capacity: int
    hold_stops: List[str]


class Route_Schema(ABC):
    """ Abstract class, served as a template for defining and managing route information.

    Various abstract methods need to be implemented in the subclass to define the route information.
    The class represents the overall topology or structure of how routes are organized and connected within a network.
    It encapsulates information about route directions, start/ending terminals, visited stops, and other relevant details.

    Attributes
        route_infos: a dictionary of Route objects that contains all the route information, i.e., {route_id -> Route}
        route_OD_rate_table: a dictionary of route-specific od rate table, i.e., {route_id -> {stop_id -> {stop_id -> od_rate}}}
        end_terminal: a dictionary of route-specific ending terminal, i.e., {route_id -> terminal_id}
        terminal_to_routes_info: a dictionary of terminal to routes, i.e., {terminal_id -> [Route]}

    Abstract Methods need to be implemented in the subclass:
        _define_route_ids(self) -> List[str]
        _define_od_table(self) -> Dict[str, Dict[str, Dict[str, float]]]
        _define_schedule_headway(self) -> Dict[str, Tuple[float, float]]
        _define_terminal(self) -> Dict[str, str]
        _define_visit_seq_stops(self) -> Dict[str, List[str]]
        _define_end_terminal(self) -> Dict[str, str]
        _define_boarding_rate(self) -> Dict[str, Dict[str, float]]

    """

    def __init__(self) -> None:
        # a list of ids of routes running in the network
        self._route_ids: List[str] = self._define_route_ids()

        # od demand rate (i.e., pax/sec) table for each route
        # route -> od_rate_table
        # od_rate_table: Dict[stop_id, Dict[stop_id, od_rate]]
        self._route_od_rate_table: Dict[str, Dict[str,
                                                  Dict[str, float]]] = self._define_od_table()

        # each route's starting terminal id
        # route -> terminal id
        self._route_terminals: Dict[str, str] = self._define_terminal()

        # each route's visited stop ids
        # route -> visited stop ids
        self._route_visit_seq_stops: Dict[str,
                                          List[str]] = self._define_visit_seq_stops()

        # each route's ending terminal id
        # route -> ending terminal id
        self._route_end_terminals: Dict[str, str] = self._define_end_terminal()

        # each route's schedule headway
        # route -> schedule (dispatching) headway
        self._route_schedule_headway_infos: Dict[str,
                                                 Tuple[float, float]] = self._define_schedule_headway()

        # each route's stop-specific boarding rate
        self._route_boarding_rate: Dict[str,
                                        Dict[str, float]] = self._define_boarding_rate()

        self._route_hold_stops: Dict[str,
                                     List[str]] = self._define_hold_stops()

        # transform the route information into a dictionary of Route objects
        self._route_details_by_id: Dict[str, Route_Details] = {}
        for route_id in self._route_ids:
            route_details = Route_Details(
                route_id=route_id,
                terminal_id=self._route_terminals[route_id],
                visit_seq_stops=self._route_visit_seq_stops[route_id],
                end_terminal_id=self._route_end_terminals[route_id],
                od_rate_table=self._route_od_rate_table[route_id],
                schedule_headway=self._route_schedule_headway_infos[route_id][0],
                schedule_headway_std=self._route_schedule_headway_infos[route_id][1],
                boarding_rate=self._route_boarding_rate[route_id],
                bus_capacity=INF,
                hold_stops=self._route_hold_stops[route_id]
            )
            self._route_details_by_id[route_id] = route_details

    @property
    def route_details_by_id(self) -> Dict[str, Route_Details]:
        return self._route_details_by_id

    @property
    def route_OD_rate_table(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        return self._route_od_rate_table

    @property
    def end_terminal(self) -> Dict[str, str]:
        return self._route_end_terminals

    @property
    def terminal_to_routes_info(self) -> Dict[str, List[Route_Details]]:
        terminal_to_route_info = {}

        # for starting terminal with dispatching routes
        terminal_to_routes = self._find_terminal_to_common_routes()
        for terminal_id, routes in terminal_to_routes.items():
            terminal_to_route_info[terminal_id] = [
                self._route_details_by_id[route] for route in routes]

        # for ending terminal without any dispatching routes
        for route, terminal in self._route_end_terminals.items():
            if terminal not in terminal_to_route_info:
                terminal_to_route_info[terminal] = []

        return terminal_to_route_info

    def _find_terminal_to_common_routes(self) -> Dict[str, List[str]]:
        terminal_to_routes = {}
        for route, terminal in self._route_terminals.items():
            if terminal not in terminal_to_routes:
                terminal_to_routes[terminal] = []
            terminal_to_routes[terminal].append(route)
        return terminal_to_routes

    @abstractmethod
    def _define_route_ids(self) -> List[str]:
        ...

    @abstractmethod
    def _define_od_table(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        ...

    @abstractmethod
    def _define_schedule_headway(self) -> Dict[str, Tuple[float, float]]:
        ...

    @abstractmethod
    def _define_terminal(self) -> Dict[str, str]:
        ...

    @abstractmethod
    def _define_visit_seq_stops(self) -> Dict[str, List[str]]:
        ...

    @abstractmethod
    def _define_end_terminal(self) -> Dict[str, str]:
        ...

    @abstractmethod
    def _define_boarding_rate(self) -> Dict[str, Dict[str, float]]:
        ...

    @abstractmethod
    def _define_hold_stops(self) -> Dict[str, List[str]]:
        ...

    # @abstractmethod
    # def _define_pax_board_info(self) -> Dict[str, Dict[str, Tuple[float, float, str]]]:
    #     ...
