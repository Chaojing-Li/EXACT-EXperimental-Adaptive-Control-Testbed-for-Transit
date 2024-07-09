from typing import Dict, Tuple, Literal
from collections import defaultdict

from .homo_one_route import Homo_One_Route_Network, Homo_One_Route_Route_Schema
from .chengdu import CD_Route3_Network, CD_Route3_Route_Schema
from .guangzhou_brt import GBRT_Network, GBRT_Route_Schema
from .network import Network
from .route import Route_Schema


class Blueprint:
    ''' Provide network and route information as a whole.

    Attributes:
        env_name: environment name
        route_schema: an object containing bus route information and some managing functions
        network: an object containing static network structure and some managing functions

    '''

    def __init__(self, env_name: str) -> None:
        self.env_name: str = env_name
        if env_name == 'homogeneous_one_route':
            self.network: Network = Homo_One_Route_Network()
            self.route_schema: Route_Schema = Homo_One_Route_Route_Schema()
        elif self.env_name == 'cd_route_3':
            self.network: Network = CD_Route3_Network()
            self.route_schema: Route_Schema = CD_Route3_Route_Schema()
        elif self.env_name == 'gbrt':
            self.network: Network = GBRT_Network()
            self.route_schema: Route_Schema = GBRT_Route_Schema()

        # _route_node_to_link: used for querying next link for current node
        # _route_link_to_node: used for querying next node for current link
        # {route_id -> {node_id -> link_id}}, {route_id -> {link_id -> node_id}}
        self._route_node_to_link, self._route_link_to_node = self._generate_node_and_link_map()

        # _route_node_distance: distance of node from the terminal
        # {route_id -> {node_id -> distance from terminal}
        self._route_node_distance = self._generate_node_distance_from_terminal()

        self._route_stop_arrival_rate = self._calculate_total_arrival_rate()

    def get_next_link_id(self, route_id: str, curr_node_id: str):
        ''' Get the next link id given the current node id of a route.

        Args:
            route_id: the route id
            curr_node_id: the current node id

        Returns:
            next_link_id: the next link id
        '''
        return self._route_node_to_link[route_id][curr_node_id]

    def get_next_node_id(self, route_id: str, curr_link_id: str) -> Tuple[str, bool]:
        ''' Get the next node id given the current link id of a route.

        Args:
            route_id: the route id
            curr_link_id: the current link id

        Returns:
            next_node_id: the next node id
        '''
        next_node_id = self._route_link_to_node[route_id][curr_link_id]
        is_ending_terminal = True if next_node_id == self.route_schema.end_terminal[
            route_id] else False
        return next_node_id, is_ending_terminal

    def get_previous_node(self, route_id: str, curr_node_id: str) -> Tuple[Literal['terminal', 'stop'], str]:
        ''' Get the previous node given the current node id of a route.

        Args:
            route_id: the route id
            curr_stop_id: the current node id

        Returns:
            node_type: the type of the previous node, either 'terminal' or 'stop'
            previous_node_id: the previous node id

        '''
        visit_seq_stops = self.route_schema.route_details_by_id[route_id].visit_seq_stops
        assert curr_node_id in visit_seq_stops, 'The query node must be a stop node'
        first_stop_id = visit_seq_stops[0]
        if curr_node_id == first_stop_id:
            terminal_id = self.route_schema.route_details_by_id[route_id].terminal_id
            return 'terminal', terminal_id
        else:
            previous_stop_index = visit_seq_stops.index(curr_node_id) - 1
            return 'stop', visit_seq_stops[previous_stop_index]

    @property
    def route_node_distance(self) -> Dict[str, Dict[str, float]]:
        ''' Return the distance from terminal to each node for each route

        Returns:
            self._route_node_distance: {route_id -> {node_id -> distance from terminal}}

        '''
        return self._route_node_distance

    @property
    def route_stop_arrival_rate(self) -> Dict[str, Dict[str, float]]:
        ''' Return the total arrival rate at each stop for each route

        Returns:
            self._route_stop_arrival_rate: {route_id -> {stop_id -> arrival rate}}

        '''
        return self._route_stop_arrival_rate

    def _generate_node_and_link_map(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
        ''' Generate the map from node to link and from link to node for each route

        The nodes include the starting and ending terminal nodes and the stop nodes.

        Returns:
            route_node_to_link: {route_id -> {node_id -> link_id}}
            route_link_to_node: {route_id -> {link_id -> node_id}}

        '''
        route_node_to_link: Dict[str, Dict[str, str]] = {}
        route_link_to_node: Dict[str, Dict[str, str]] = {}

        for route_id, route in self.route_schema.route_details_by_id.items():
            node_to_link: Dict[str, str] = {}
            link_to_node: Dict[str, str] = {}
            node_seqs = [route.terminal_id] + \
                route.visit_seq_stops + [route.end_terminal_id]

            for head_node, tail_node in zip(node_seqs[:-1], node_seqs[1:]):
                link_id = self.network.get_link_id_by_two_nodes(
                    head_node, tail_node)
                node_to_link[head_node] = link_id
                link_to_node[link_id] = tail_node

            route_node_to_link[route_id] = node_to_link
            route_link_to_node[route_id] = link_to_node
        return route_node_to_link, route_link_to_node

    def _generate_node_distance_from_terminal(self) -> Dict[str, Dict[str, float]]:
        ''' Generate the distance from terminal to each node for each route

        The nodes include the starting and ending terminal nodes and the stop nodes.

        Returns:
            route_node_distance: {route_id -> {node_id -> distance from terminal}}

        '''
        route_node_distance: Dict[str, Dict[str, float]] = {}
        for route_id, route in self.route_schema.route_details_by_id.items():
            distance_cum = 0
            node_distance = {}
            node_seq = [route.terminal_id] + \
                route.visit_seq_stops + [route.end_terminal_id]
            for head_node, tail_node in zip(node_seq[0:-1], node_seq[1:]):
                node_distance[head_node] = distance_cum
                distance = self._get_distance(head_node, tail_node)
                distance_cum += distance

            node_distance[route.end_terminal_id] = distance_cum
            route_node_distance[route_id] = node_distance
        return route_node_distance

    def _get_distance(self, node_1_id, node_2_id):
        ''' Get the travel distance for two nodes

        Args:
            node_1_id: the first node id
            node_2_id: the second node id

        Returns:
            distance: the travel distance between the two nodes
        # TODO for now simply use manhattan distance, should be replaced by real travel distance

        '''
        node_1_x, node_1_y = self.network.get_node_xy(node_1_id)
        node_2_x, node_2_y = self.network.get_node_xy(node_2_id)
        distance = abs(node_1_x - node_2_x) + abs(node_1_y - node_2_y)
        return distance

    def _calculate_total_arrival_rate(self) -> Dict[str, Dict[str, float]]:
        ''' Calculate the total arrival rate at each stop for each route by summing up the OD table by row.
        '''
        route_total_arrival_rate = defaultdict(dict)
        for route_id, route in self.route_schema.route_details_by_id.items():
            for origin_stop_id, destination_rate in route.od_rate_table.items():
                total_origin_demand = sum(destination_rate.values())
                route_total_arrival_rate[route_id][origin_stop_id] = total_origin_demand

            last_stop_id = route.visit_seq_stops[-1]

            # # case 1. the last stop's arrival demand rate is 0, i.e., no one will get on the bus at the last stop
            route_total_arrival_rate[route_id][last_stop_id] = 0.0

            # case 2. the last stop's arrival demand rate equals the last but one stop's arrival demand rate
            # last_but_one_stop_id = route.visit_seq_stops[-2]
            # route_total_arrival_rate[route_id][last_stop_id] = route_total_arrival_rate[route_id][last_but_one_stop_id]

        return dict(route_total_arrival_rate)
