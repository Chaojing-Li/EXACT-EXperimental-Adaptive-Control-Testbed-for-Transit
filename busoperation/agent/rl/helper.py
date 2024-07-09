from torch_geometric.data import Data
import torch
from typing import Dict, Tuple, Optional, List
from .rl_dataclass import Event
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_undirected
import networkx as nx
import matplotlib.pyplot as plt
import time
import functools


def find_for_and_backward_buses(bus_id_loc: Dict[str, float],
                                curr_bus_id: str
                                ) -> Tuple[Optional[str], float, Optional[str], float]:
    ''' Find the forward and backward buses and spacings of the current bus

    '''
    curr_loc = bus_id_loc[curr_bus_id]

    greater_bus_id = None
    greater_loc_diff = float('inf')
    smaller_bus_id = None
    smaller_loc_diff = float('inf')

    for bus_id, loc in bus_id_loc.items():
        if bus_id != curr_bus_id:
            loc_diff = loc - curr_loc
            if loc_diff > 0 and loc_diff < greater_loc_diff:
                greater_bus_id = bus_id
                greater_loc_diff = loc_diff
            elif loc_diff < 0 and abs(loc_diff) < smaller_loc_diff:
                smaller_bus_id = bus_id
                smaller_loc_diff = abs(loc_diff)

    return greater_bus_id, greater_loc_diff, smaller_bus_id, smaller_loc_diff


def construct_graph(curr_bus_id: str,
                    events: List[Event],
                    visit_seq_stops: Tuple[str, ...],
                    curr_stop_id: str,
                    next_stop_id: str,
                    curr_time: int,
                    next_time: int) -> Tuple[Optional[Data], Optional[Data]]:
    filtered_events = [
        event for event in events if curr_time < event.time < next_time]

    self_events = [event for event in events if event.time ==
                   curr_time and event.stop_id == curr_stop_id and event.bus_id == curr_bus_id]

    # assert len(self_events) == 1, 'must be only one event'
    self_event = self_events[0]

    downstream_events = []
    upstream_events = []
    curr_stop_idx = visit_seq_stops.index(curr_stop_id)
    for event in filtered_events:
        if event.stop_id in visit_seq_stops[0:curr_stop_idx+1]:
            upstream_events.append(event)
        else:
            downstream_events.append(event)

    upstream_graph, downstream_graph = None, None
    if len(upstream_events) > 0 and len(downstream_events) > 0:
        # node features
        up_xs, down_xs = [], []
        self_xs = [s for s in self_event.state]
        self_xs.append(self_event.action)
        up_xs.append(self_xs)
        down_xs.append(self_xs)

        for event in upstream_events:
            # xs = [s for s in event.state]
            xs = [s if s != float('inf') else -1 for s in event.state]
            xs.append(event.action)
            up_xs.append(xs)
        up_xs = torch.tensor(up_xs, dtype=torch.float)

        for event in downstream_events:
            # xs = [s for s in event.state]
            xs = [s if s != float('inf') else -1 for s in event.state]
            xs.append(event.action)
            down_xs.append(xs)
        down_xs = torch.tensor(down_xs, dtype=torch.float)

        # edge connectivity
        source_nodes = list(range(1, len(upstream_events)+1))
        target_nodes = [0] * len(upstream_events)

        # 1. one direction
        # up_edge_index = [source_nodes, target_nodes]

        # 2. bidirectional
        up_edge_index = [source_nodes, target_nodes]
        up_edge_index = torch.tensor(up_edge_index, dtype=torch.long)
        # up_edge_index = to_undirected(up_edge_index)

        source_nodes = list(range(1, len(downstream_events)+1))
        target_nodes = [0] * len(downstream_events)

        # 1. one direction
        # down_edge_index = [source_nodes, target_nodes]
        # 2. bidirectional
        down_edge_index = [source_nodes, target_nodes]
        down_edge_index = torch.tensor(down_edge_index, dtype=torch.long)
        # down_edge_index = to_undirected(down_edge_index)

        upstream_graph = Data(x=up_xs, edge_index=up_edge_index)
        downstream_graph = Data(x=down_xs, edge_index=down_edge_index)
        # vis = to_networkx(downstream_graph)
        # nx.draw(vis, with_labels=True)
        # plt.show()

    return upstream_graph, downstream_graph


def time_func(func):
    """timefunc's doc"""

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        """time_wrapper's doc string"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(f"Function: {func.__name__}, Time: {time_elapsed}")
        return result

    return time_closure
