from collections import defaultdict
from typing import List, Dict, Tuple
from typing_extensions import override


from .chengdu_route_3_data.dataloader import DataLoader
from .network import Network
from .route import Route_Schema
from .config_dataclass import *
from .utils import print_od_table, sum_entries, sum_entries_with_row_sums


data_loader = DataLoader()
node_ids = data_loader.node_ids
link_time_info = data_loader.link_time_info
link_spacing = data_loader.spacing
stop_pax_arrival_rate = data_loader.stop_pax_arrival_rate
H_mean, H_std = data_loader.dispatching_headway
H_mean = 300

berth_num = 3
start_terminal_id = node_ids[0]
end_terminal_id = node_ids[-1]
visit_seq_stop_ids = node_ids[1:-1]
# print('start_terminal_id:', start_terminal_id)
# print('end_terminal_id:', end_terminal_id)
# print('visit_seq_stop_ids:', visit_seq_stop_ids)

# the cumulative x coordinate of each node, i.e., distance from the start terminal
node_x_cum = {}
x_cum = 0
node_x_cum[start_terminal_id] = x_cum
for head_node, tail_node in zip(node_ids[:-1], node_ids[1:]):
    spacing = link_spacing[tail_node]
    x_cum += spacing
    node_x_cum[tail_node] = x_cum


class CD_Route3_Network(Network):
    def __init__(self) -> None:
        super().__init__()

    @override
    def _define_network(self):
        y = 0
        # build nodes
        for node, x in node_x_cum.items():
            if node == start_terminal_id or node == end_terminal_id:
                terminal_node_geometry = TerminalNodeGeometry(x, y)
                self._G.add_node(node, node_type='terminal',
                                 terminal_node_geometry=terminal_node_geometry)
            else:
                stop_node_geometry = StopNodeGeometry(x, y, berth_num)
                self._G.add_node(node, node_type='stop',
                                 stop_node_geometry=stop_node_geometry)
            self._name_coordinates[node] = (x, y)

        # build links
        head_x_cum = 0
        link_id = 0
        for head_node, tail_node in zip(node_ids[:-1], node_ids[1:]):
            spacing = link_spacing[tail_node]

            tt_mean = link_time_info[tail_node]['loc'] + 23.2
            tt_cv = link_time_info[tail_node]['scale'] / tt_mean
            tt_cv *= 0.922

            tt_type = 'normal'
            link_distribution = LinkDistribution(tt_mean, tt_cv, tt_type)
            link_geometry = LinkGeometry(str(head_node), str(
                tail_node), head_x_cum, 0, spacing)
            self._G.add_edge(str(head_node), str(tail_node), link_id=link_id,
                             link_geometry=link_geometry, link_distribution=link_distribution)
            link_id += 1
            head_x_cum += spacing


class CD_Route3_Route_Schema(Route_Schema):
    def __init__(self) -> None:
        super().__init__()

    @override
    def _define_od_table(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        # we assume that alighting follows a uniform distribution
        od_rate_table = defaultdict(dict)
        for idx, origin_stop in enumerate(visit_seq_stop_ids):
            pax_arrival_rate = stop_pax_arrival_rate[origin_stop]
            dest_stops = visit_seq_stop_ids[idx+1:]
            dest_stop_num = len(dest_stops)
            if len(dest_stops) > 0:
                for dest_stop in dest_stops:
                    od_rate_table[origin_stop][dest_stop] = pax_arrival_rate * \
                        1.1 / dest_stop_num
            else:
                # the final visited stop does not have no od rate
                for dest_stop in visit_seq_stop_ids:
                    od_rate_table[origin_stop][dest_stop] = 0.0
        od_rate_table = dict(od_rate_table)
        # print_od_table(od_rate_table, visit_seq_stop_ids)
        return {'3': od_rate_table}

    @override
    def _define_route_ids(self) -> List[str]:
        return ['3']

    @override
    def _define_schedule_headway(self) -> Dict[str, Tuple[float, float]]:
        # return {'0': (H_mean, H_std)}
        return {'3': (H_mean, 0)}

    @override
    def _define_terminal(self) -> Dict[str, str]:
        return {'3': start_terminal_id}

    @override
    def _define_visit_seq_stops(self) -> Dict[str, List[str]]:
        return {'3': visit_seq_stop_ids}

    @override
    def _define_end_terminal(self) -> Dict[str, str]:
        return {'3': end_terminal_id}

    @override
    def _define_boarding_rate(self) -> Dict[str, Dict[str, float]]:
        return {'3': {stop_id: 1/4.0 for stop_id in visit_seq_stop_ids}}

    @override
    def _define_hold_stops(self) -> Dict[str, List[str]]:
        hold_stops = visit_seq_stop_ids.copy()
        return {'3': hold_stops}

    # def _define_od_table(self) -> Dict[str, Dict[str, Dict[str, float]]]:
    #     od_rate_table = defaultdict(dict)
    #     for idx, origin_stop in enumerate(visit_seq_stop_ids):
    #         for dest_stop in visit_seq_stop_ids[idx+1:]:
    #             # all the passengers will be dropped off at the last stop
    #             if dest_stop != visit_seq_stop_ids[-1]:
    #                 od_rate_table[origin_stop][dest_stop] = 0.0
    #             else:
    #                 # od_rate_table[origin_stop][dest_stop] = stop_pax_arrival_rate[origin_stop] * 1.2
    #                 od_rate_table[origin_stop][dest_stop] = 1.0/60

    #     od_rate_table = dict(od_rate_table)
    #     return {'0': od_rate_table}
