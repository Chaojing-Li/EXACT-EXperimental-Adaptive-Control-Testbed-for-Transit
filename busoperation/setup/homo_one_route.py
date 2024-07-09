from collections import defaultdict
from typing import List, Dict, Tuple
from typing_extensions import override

from .config_dataclass import *
from .network import Network
from .route import Route_Schema

# stop number excluding starting and ending stop
stop_num = 35
berth_num = 3
H = 300.0  # seconds
start_terminal = str(0)
end_terminal = str(stop_num+1)
visit_seq_stops = [str(x) for x in range(1, stop_num+1)]

spacing = 1000  # meters
speed = 20.0  # m/s
tt_mean = spacing / speed
tt_cv = 0.1
tt_type = 'normal'
pax_arrival_rate = 4.0 / 60  # pax per second

corridor_length = (stop_num+1) * spacing


class Homo_One_Route_Network(Network):
    def __init__(self) -> None:
        super().__init__()

    def _define_network(self):
        x_cum = 0
        y_cum = 0

        # build nodes
        for node_id in range(stop_num+2):
            x = x_cum
            y = y_cum
            node_id = str(node_id)
            if node_id == start_terminal or node_id == end_terminal:
                terminal_node_geometry = TerminalNodeGeometry(x, y)
                self._G.add_node(node_id, node_type='terminal',
                                 terminal_node_geometry=terminal_node_geometry)
            else:
                # stop_node_geometry = StopNodeGeometry(x, y, berth_num, x_cum)
                stop_node_geometry = StopNodeGeometry(x, y, berth_num)
                self._G.add_node(node_id, node_type='stop',
                                 stop_node_geometry=stop_node_geometry)
            self._name_coordinates[node_id] = (x, y)
            x_cum += spacing

        # build links
        # homogeneous link operation distribution
        link_distribution = LinkDistribution(tt_mean, tt_cv, tt_type)

        x_cum = 0
        y_cum = 0
        # link - terminal to first stop
        link_id = 0
        link_geometry = LinkGeometry(
            start_terminal, visit_seq_stops[0], x_cum, y_cum, spacing)
        self._G.add_edge(start_terminal, visit_seq_stops[0], link_id=link_id,
                         link_geometry=link_geometry, link_distribution=link_distribution)
        link_id += 1
        x_cum += spacing

        # link - between stops
        for curr_stop, next_stop in zip(visit_seq_stops[0:-1], visit_seq_stops[1:]):
            link_geometry = LinkGeometry(
                curr_stop, next_stop, x_cum, y_cum, spacing)

            self._G.add_edge(curr_stop, next_stop, link_id=link_id,
                             link_geometry=link_geometry, link_distribution=link_distribution)
            link_id += 1
            x_cum += spacing

        # link - last stop to ending terminal
        link_geometry = LinkGeometry(
            str(visit_seq_stops[-1]), str(end_terminal), x_cum, y_cum, spacing)
        self._G.add_edge(str(visit_seq_stops[-1]), str(end_terminal), link_id=link_id,
                         link_geometry=link_geometry, link_distribution=link_distribution)


class Homo_One_Route_Route_Schema(Route_Schema):
    def __init__(self) -> None:
        super().__init__()

    def _define_od_table(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        od_rate_table = defaultdict(dict)
        for idx, origin_stop in enumerate(visit_seq_stops):
            for dest_stop in visit_seq_stops[idx+1:]:
                if dest_stop != str(stop_num):
                    od_rate_table[origin_stop][dest_stop] = 0.0
                else:
                    od_rate_table[origin_stop][dest_stop] = pax_arrival_rate
        od_rate_table = dict(od_rate_table)
        return {'0': od_rate_table}

    @override
    def _define_route_ids(self) -> List[str]:
        return ['0']

    @override
    def _define_schedule_headway(self) -> Dict[str, Tuple[float, float]]:
        return {'0': (H, 0)}

    @override
    def _define_terminal(self) -> Dict[str, str]:
        return {'0': start_terminal}

    @override
    def _define_visit_seq_stops(self) -> Dict[str, List[str]]:
        return {'0': visit_seq_stops}

    @override
    def _define_end_terminal(self) -> Dict[str, str]:
        return {'0': end_terminal}

    @override
    def _define_boarding_rate(self) -> Dict[str, Dict[str, float]]:
        return {'0': {stop_id: 1/2.0 for stop_id in visit_seq_stops}}

    @override
    def _define_hold_stops(self) -> Dict[str, List[str]]:
        return {'0': visit_seq_stops}

    # @override
    # def _define_pax_board_info(self) -> Dict[str, Dict[str, Tuple[float, float, str]]]:
    #     board_time_mean = 2.0
    #     board_time_std = 2.0*0.25
    #     board_time_type = 'normal'
    #     return {'0': {stop_id: (board_time_mean, board_time_std, board_time_type) for stop_id in visit_seq_stops}}
