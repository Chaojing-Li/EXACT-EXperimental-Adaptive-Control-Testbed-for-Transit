from abc import ABC, abstractmethod
from typing import TypeVar, Dict, Tuple

import networkx as nx
import matplotlib.pyplot as plt

from setup.config_dataclass import StopNodeGeometry, TerminalNodeGeometry, LinkGeometry, LinkDistribution
T = TypeVar("T", bound="Network")


class Network(ABC):
    ''' Abstract class for defining the network structure with the help of `networkx`.

    Attributes:
        terminal_node_geometry_info: a dictionary with key as terminal node id and value as terminal node geometry information
        stop_node_geometry_info: a dictionary with key as stop node id and value as stop node geometry information
        link_geometry_info: a dictionary with key as link id and value as link geometry information
        link_distribution: a dictionary with key as link id and value as link distribution information

    Methods:
        get_link_id_by_two_nodes(self, head_node: str, tail_node: str) -> str
        get_node_xy(self, node_id) -> Tuple[float, float]
        visualize(self) -> None

    Abstract Methods:
        _define_network(self) -> None

    '''

    def __init__(self) -> None:
        # a directed graph representing the network using `networkx` package
        self._G: nx.DiGraph = nx.DiGraph()
        # a dictionary with key as node name and value as node coordinates. It is used for visualization.
        self._name_coordinates: Dict[str, Tuple[float, float]] = {}
        # inherited by the user to define the network structure
        self._define_network()

    def get_link_id_by_two_nodes(self, head_node: str, tail_node: str) -> str:
        ''' Return the link id by two nodes.

        Args:
            head_node: the head node of the link
            tail_node: the tail node of the link

        Returns:
            link_id: id of the link
        '''
        edge = self._G.edges[head_node, tail_node]
        return str(edge['link_id'])

    # def get_next_link_id(self, curr_stop_id: str, next_stop_id: str) -> str:
    #     edge = self._G.edges[str(curr_stop_id), str(next_stop_id)]
    #     return str(edge['link_id'])

    def get_node_xy(self, node_id):
        ''' Get the x, y coordinates of a node.

        Args:
            node_id: id of the node

        Returns:
            x, y: x and y coordinates of the node

        '''
        node = self._G.nodes[node_id]
        if node['node_type'] == 'terminal':
            x, y = node['terminal_node_geometry'].x, node['terminal_node_geometry'].y
        else:
            x, y = node['stop_node_geometry'].x, node['stop_node_geometry'].y
        return x, y

    @property
    def terminal_node_geometry_info(self) -> Dict[str, TerminalNodeGeometry]:
        ''' Get the terminal node geometry information.

        Returns:
            terminal_node_geometry_info: a dictionary with key as terminal node id and value as terminal node geometry information

        '''
        terminal_node_geometry_info = {}
        for node, info in self._G.nodes.items():
            if info['node_type'] == 'terminal':
                terminal_node_geometry_info[node] = info['terminal_node_geometry']
        return terminal_node_geometry_info

    @property
    def stop_node_geometry_info(self) -> Dict[str, StopNodeGeometry]:
        ''' Get the stop node geometry information.

        Returns:
            stop_node_geometry_info: a dictionary with key as stop node id and value as stop node geometry information

        '''
        stop_node_geometry_info = {}
        for node, info in self._G.nodes.items():
            if info['node_type'] == 'stop':
                stop_node_geometry_info[node] = info['stop_node_geometry']
        return stop_node_geometry_info

    @property
    def link_geometry_info(self) -> Dict[str, LinkGeometry]:
        ''' Get the link geometry information.

        Returns:
            link_geometry_info: a dictionary with key as link id and value as link geometry information

        '''
        link_geometry_info = {}
        for edge in self._G.edges:
            edge_data = self._G.get_edge_data(edge[0], edge[1])
            link_id = str(edge_data['link_id'])
            link_geometry_info[link_id] = edge_data['link_geometry']
        return link_geometry_info

    @property
    def link_distribution(self) -> Dict[str, LinkDistribution]:
        link_distribution = {}
        for edge in self._G.edges:
            edge_data = self._G.get_edge_data(edge[0], edge[1])
            link_id = str(edge_data['link_id'])
            link_distribution[link_id] = edge_data['link_distribution']
        return link_distribution

    def visualize(self) -> None:
        # fig, ax = plt.subplots()
        # # draw the nodes
        # pos = self._name_coordinates
        # # colors = ['r' if node[1]['type'] ==
        # #           'terminal' else 'b' for node in self._G.nodes.data()]
        # # nx.draw_networkx_nodes(self._G, pos, node_color=colors, ax=ax)
        # nx.draw_networkx_nodes(self._G, pos, ax=ax)
        # nx.draw_networkx_labels(self._G, pos, ax=ax)

        # # draw the edges
        # nx.draw_networkx_edges(self._G, pos, ax=ax, edgelist=self._G.edges)
        # edge_weights = nx.get_edge_attributes(self._G, 'link_id')
        # edge_labels = {edge: edge_weights[edge] for edge in self._G.edges}
        # nx.draw_networkx_edge_labels(
        #     self._G, pos, ax=ax, edge_labels=edge_labels)
        # plt.show()

        fig, ax = plt.subplots()  # Increase the figure size
        pos = self._name_coordinates
        # Increase the node size
        node_size = 400
        nx.draw_networkx_nodes(self._G, pos, ax=ax, node_size=node_size)

        # Adjust the font size and properties for node labels
        node_font_size = 12
        node_font_family = 'sans-serif'
        node_font_weight = 'bold'

        # Calculate the vertical offset for node labels
        vertical_offset = 10.0  # Adjust this value to control the distance above the nodes
        label_pos = {node: (coords[0], coords[1] + vertical_offset)
                     for node, coords in pos.items()}

        nx.draw_networkx_labels(self._G, label_pos, ax=ax, font_size=node_font_size,
                                font_family=node_font_family, font_weight=node_font_weight,
                                verticalalignment='bottom')

        # Draw the edges
        nx.draw_networkx_edges(self._G, pos, ax=ax, edgelist=self._G.edges)

        # Adjust the font size and properties for edge labels
        edge_font_size = 10
        edge_font_family = 'sans-serif'
        edge_weights = nx.get_edge_attributes(self._G, 'link_id')
        edge_labels = {edge: edge_weights[edge] for edge in self._G.edges}
        nx.draw_networkx_edge_labels(self._G, pos, ax=ax, edge_labels=edge_labels,
                                     font_size=edge_font_size, font_family=edge_font_family)

        # Optionally, remove axis labels and ticks
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    # Doing this ensures that subclasses of Network will
    # return the correct type when calling `define_network`.
    @abstractmethod
    def _define_network(self):
        ''' Subclass must implement this method to define the network structure.

                All the nodes and links should be added to the graph `_G`, 
                    and the node coordinates should be added to the dictionary `_name_coordinates`.

                '''
        ...
