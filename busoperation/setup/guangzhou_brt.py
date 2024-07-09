from collections import defaultdict
from typing import List, Dict, Tuple
from typing_extensions import override

import numpy as np
import pandas as pd

from .guangzhou_brt_data.dataloader import DataLoader

from .network import Network
from .route import Route_Schema
from .config_dataclass import *
from .utils import print_od_table, sum_entries, sum_entries_with_row_sums


class GBRT_Network(Network):
    def __init__(self) -> None:
        self.data_loader = DataLoader()
        super().__init__()

    @override
    def _define_network(self):

        speed = 30/3.6

        stop_names = ['DPZ', 'CB', 'TLMJ', 'TD',
                      'TX', 'XY', 'SS', 'HJXC', 'SDJD', 'GD']
        link_travel_means = [53.1, 58.1, 24.2,
                             32.5, 102.3, 35.5, 69.6, 90.6, 87.5]
        link_spacings = [speed * x for x in link_travel_means]
        link_travel_stds = [11.3, 22.5, 9.5, 8.5, 24.7, 8.5, 24.0, 25.5, 41.5]
        # link_travel_stds = [x for x in link_travel_stds]

        offset = 500
        y_offset = 150
        # build terminals
        # start_terminal_1 controls most bus lines' arrival (upstream_DPZ)
        # start_terminal_2 controls line B21's arrival (upstream_TD)
        # end_terminal_1 controls most bus lines' departure (downstream_GD)
        # end_terminal_2 controls B16 and B20's departure (downstream_SDJD)
        # end_terminal_3 controls B19's departure (downstream_DPZ)
        terminal_infos = [('upstream_DPZ', 0, 0),
                          ('downstream_GD', sum(link_spacings)+2*offset, 0),
                          ('upstream_TD', sum(
                              link_spacings[0:3])+offset, y_offset),
                          ('downstream_SDJD', sum(
                              link_spacings[0:8])+offset, -y_offset),
                          ('downstream_DPZ', offset, y_offset)
                          ]
        for terminal_name, terminal_x, terminal_y in terminal_infos:
            terminal_node_geometry = TerminalNodeGeometry(
                terminal_x, terminal_y)
            self._G.add_node(terminal_name, node_type='terminal',
                             terminal_node_geometry=terminal_node_geometry)
            self._name_coordinates[terminal_name] = (terminal_x, terminal_y)

        # build stops
        stop_infos = [('DPZ', offset, 0),
                      ('CB', offset+link_spacings[0], 0),
                      ('TLMJ', offset+sum(link_spacings[0:2]), 0),
                      ('TD', offset+sum(link_spacings[0:3]), 0),
                      ('TX', offset+sum(link_spacings[0:4]), 0),
                      ('XY', offset+sum(link_spacings[0:5]), 0),
                      ('SS', offset+sum(link_spacings[0:6]), 0),
                      ('HJXC', offset+sum(link_spacings[0:7]), 0),
                      ('SDJD', offset+sum(link_spacings[0:8]), 0),
                      ('GD', offset+sum(link_spacings[0:9]), 0),
                      ]
        for stop_name, stop_x, stop_y in stop_infos:
            stop_node_geometry = StopNodeGeometry(stop_x, stop_y, 3)
            self._G.add_node(stop_name, node_type='stop',
                             stop_node_geometry=stop_node_geometry)
            self._name_coordinates[stop_name] = (stop_x, stop_y)

        # build links
        link_infos = [('upstream_DPZ', 'DPZ', offset,
                       offset/speed, 0, 'normal'),
                      ('DPZ', 'CB', link_spacings[0], link_travel_means[0],
                       link_travel_stds[0], 'normal'),
                      ('CB', 'TLMJ', link_spacings[1], link_travel_means[1],
                       link_travel_stds[1], 'normal'),
                      ('TLMJ', 'TD', link_spacings[2], link_travel_means[2],
                       link_travel_stds[2], 'normal'),
                      ('TD', 'TX', link_spacings[3], link_travel_means[3],
                       link_travel_stds[3], 'normal'),
                      ('TX', 'XY', link_spacings[4], link_travel_means[4],
                       link_travel_stds[4], 'normal'),
                      ('XY', 'SS', link_spacings[5], link_travel_means[5],
                       link_travel_stds[5], 'normal'),
                      ('SS', 'HJXC', link_spacings[6], link_travel_means[6],
                       link_travel_stds[6], 'normal'),
                      ('HJXC', 'SDJD', link_spacings[7], link_travel_means[7],
                       link_travel_stds[7], 'normal'),
                      ('SDJD', 'GD', link_spacings[8], link_travel_means[8],
                       link_travel_stds[8], 'normal'),
                      ('GD', 'downstream_GD', offset, offset/speed, 0, 'normal'),
                      ]

        # between start_terminal_2 and TD
        start_terminal_2_x = sum(link_spacings[0:3])+offset
        start_terminal_2_y = y_offset
        TD_x = offset+sum(link_spacings[0:3])
        TD_y = 0
        spacing = abs(TD_x - start_terminal_2_x) + \
            abs(TD_y - start_terminal_2_y)
        link_infos.append(('upstream_TD', 'TD', spacing,
                          spacing/speed, 0, 'normal'))

        # between SDJD and end_terminal_2
        end_terminal_2_x = sum(link_spacings[0:8])+offset
        end_terminal_2_y = -y_offset
        SDJD_x = offset+sum(link_spacings[0:8])
        SDJD_y = 0
        spacing = abs(SDJD_x - end_terminal_2_x) + \
            abs(SDJD_y - end_terminal_2_y)
        link_infos.append(('SDJD', 'downstream_SDJD', spacing,
                          spacing/speed, 0, 'normal'))

        # between DPZ and end_terminal_3
        end_terminal_3_x = offset
        end_terminal_3_y = y_offset
        DPZ_x = offset
        DPZ_y = 0
        spacing = abs(DPZ_x - end_terminal_3_x) + \
            abs(DPZ_y - end_terminal_3_y)
        link_infos.append(('DPZ', 'downstream_DPZ', spacing,
                          spacing/speed, 0, 'normal'))

        link_id = 0
        for head_node, tail_node, spacing, tt_mean, tt_std, tt_type in link_infos:
            link_distribution = LinkDistribution(
                tt_mean, tt_std/tt_mean, tt_type)
            link_geometry = LinkGeometry(str(head_node), str(
                tail_node), 0, 0, spacing)
            self._G.add_edge(str(head_node), str(tail_node), link_id=link_id,
                             link_geometry=link_geometry, link_distribution=link_distribution)
            link_id += 1

        # self.visualize()


class GBRT_Route_Schema(Route_Schema):
    def __init__(self) -> None:
        self.data_loader = DataLoader()
        super().__init__()

        # 'B2', 'B2A', 'B3', 'B5/B5K', 'B16', 'B20' are the main 6 routes
        # 'B21' joins the corridor at 'TD' stop
        # 'B19' visits the first stop (i.e. 'DPZ') and then leaves the corridor
        # 'B16' and 'B20' leaves corridor at 'SDJD' stop

    @override
    def _define_route_ids(self) -> List[str]:
        routes = ['B2', 'B2A', 'B3', 'B5/B5K', 'B16', 'B20', 'B19', 'B21']
        return routes

    @override
    def _define_schedule_headway(self) -> Dict[str, Tuple[float]]:
        route_schedule_headway = {}
        for route_id, H_mean in self.data_loader.dispatch_headway_mean.items():
            H_std = self.data_loader.dispatch_headway_cv[route_id] * H_mean
            route_schedule_headway[route_id] = (H_mean, H_std)
        return route_schedule_headway

    @override
    def _define_terminal(self) -> Dict[str, str]:
        route_terminal = {}
        for route_id in ['B2', 'B2A', 'B3', 'B5/B5K', 'B16', 'B20', 'B19']:
            route_terminal[route_id] = 'upstream_DPZ'
        route_terminal['B21'] = 'upstream_TD'
        return route_terminal

    @override
    def _define_end_terminal(self) -> Dict[str, str]:
        route_end_terminal = {}
        for route_id in ['B2', 'B2A', 'B3', 'B5/B5K', 'B21']:
            route_end_terminal[route_id] = 'downstream_GD'

        for route_id in ['B16', 'B20']:
            route_end_terminal[route_id] = 'downstream_SDJD'

        route_end_terminal['B19'] = 'downstream_DPZ'
        return route_end_terminal

    @override
    def _define_visit_seq_stops(self) -> Dict[str, List[str]]:
        route_visit_seq_stops = {}
        for route_id in ['B2', 'B2A', 'B3', 'B5/B5K']:
            visit_seq_stops = ['DPZ', 'CB', 'TLMJ', 'TD',
                               'TX', 'XY', 'SS', 'HJXC', 'SDJD', 'GD']
            route_visit_seq_stops[route_id] = visit_seq_stops

        for route_id in ['B16', 'B20']:
            visit_seq_stops = ['DPZ', 'CB', 'TLMJ', 'TD',
                               'TX', 'XY', 'SS', 'HJXC', 'SDJD']
            route_visit_seq_stops[route_id] = visit_seq_stops

        route_visit_seq_stops['B21'] = [
            'TD', 'TX', 'XY', 'SS', 'HJXC', 'SDJD', 'GD']

        route_visit_seq_stops['B19'] = ['DPZ']

        # self._route_visit_seq_stops = route_visit_seq_stops.copy()

        return route_visit_seq_stops

    @override
    def _define_boarding_rate(self) -> Dict[str, Dict[str, float]]:
        route_boarding_rate = {}
        for route_id in ['B2', 'B2A', 'B3', 'B5/B5K', 'B16', 'B20', 'B19', 'B21']:
            boarding_rate = {
                stop_id: 1/4.0 for stop_id in self._route_visit_seq_stops[route_id]}
            route_boarding_rate[route_id] = boarding_rate
        return route_boarding_rate

    @override
    def _define_hold_stops(self) -> Dict[str, List[str]]:
        route_hold_stops = {}
        for route_id in ['B2', 'B2A', 'B3', 'B5/B5K', 'B16', 'B20', 'B19', 'B21']:
            hold_stops = self._route_visit_seq_stops[route_id].copy()
            route_hold_stops[route_id] = hold_stops

        return route_hold_stops

    @override
    def _define_od_table(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        route_od_rate_table = {}
        route_visit_seq_stops = self._define_visit_seq_stops()
        for route_id, visit_seq_stop_ids in route_visit_seq_stops.items():
            od_rate_table = defaultdict(dict)
            for idx, origin_stop in enumerate(visit_seq_stop_ids):
                for dest_stop in visit_seq_stop_ids[idx+1:]:
                    # all the passengers will be dropped off at the last stop
                    if dest_stop != visit_seq_stop_ids[-1]:
                        od_rate_table[origin_stop][dest_stop] = 0.0
                    else:
                        od_rate_table[origin_stop][dest_stop] = 6.0/60

            route_od_rate_table[route_id] = dict(od_rate_table)

        return route_od_rate_table
