from copy import deepcopy
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
import random
import numpy as np
import torch
from torch_geometric.data import Data

from simulator.snapshot import Snapshot
from setup.blueprint import Blueprint
from simulator.virtual_bus import VirtualBus
from simulator.simulator import Simulator

from .rl_agent import RLAgent
from .net import Actor_Net, Critic_Net
from .rl_dataclass import Event
from .helper import construct_graph


@dataclass(frozen=True)
class SARS_Graph:
    state: List[float]
    action: float
    reward: Optional[float]
    next_state: List[float]
    upstream_graph: Data
    downstream_graph: Data
    next_upstream_graph: Data
    next_downstream_graph: Data


@dataclass(frozen=True)
class SAR_Graph:
    state: List[float]
    action: float
    reward: Optional[float]
    upstream_graph: Data
    downstream_graph: Data


class Local_Spacing_DDPG(RLAgent):
    def __init__(self, agent_config: Dict[str, Any], blueprint: Blueprint) -> None:
        super().__init__(agent_config, blueprint)

        self._blueprint = blueprint
        self._max_hold_time = agent_config['max_hold_time']

        # {(route_id, bus_id) -> [(stop_id, event)]}
        # this property will be dynamically deleted once processed and pushed into buffer
        self._bus_stop_events: Dict[Tuple[str, str],
                                    List[Tuple[str, Event]]] = defaultdict(list)
        # this property will only be increased until reset, used for constructing graph
        self._total_events: List[Event] = []
        self._add_event_count = 0

        # used for training
        self._state_size = agent_config['state_size']
        self._replay_buffer = deque(maxlen=agent_config['memory_size'])
        self._actor_net = Actor_Net(
            state_size=self._state_size, hidde_size=tuple(agent_config['hidden_size']))
        self._critic_net = Critic_Net(
            state_size=self._state_size, hidde_size=tuple(agent_config['hidden_size']))
        self._target_actor_net = deepcopy(self._actor_net)
        self._target_critic_net = deepcopy(self._critic_net)
        for param in self._target_actor_net.parameters():
            param.requires_grad = False
        for param in self._target_critic_net.parameters():
            param.requires_grad = False
        self._actor_optim = torch.optim.Adam(
            self._actor_net.parameters(), lr=agent_config['actor_lr'])
        self._critic_optim = torch.optim.Adam(
            self._critic_net.parameters(), lr=agent_config['critic_lr'])
        self._gamma = agent_config['gamma']
        self._polya = agent_config['polya']
        self._update_cycle = agent_config['update_cycle']
        self._batch_size = agent_config['batch_size']
        self._init_noise_level = agent_config['init_noise_level']
        self._decay_rate = agent_config['decay_rate']
        self._noise_level = self._init_noise_level

        self._learn_count = 0

    def calculate_hold_time(self, snapshot: Snapshot):
        stop_bus_hold_time = {}
        for (stop_id, route_id, bus_id) in snapshot.holder_snapshot.action_buses:

            if not snapshot.bus_snapshots[(route_id, bus_id)].is_need_to_hold:
                stop_bus_hold_time[(stop_id, route_id, bus_id)] = 0
                continue

            _, forward_spacing, _, backward_spacing = self.extract_local_info_from_snapshot(
                bus_id, snapshot, ['spacing'])
            locs = self.extract_global_info_from_snapshot(snapshot, [
                'loc'])

            forward_spacing = forward_spacing / 1000 if forward_spacing != float(
                'inf') else forward_spacing
            backward_spacing = backward_spacing / \
                1000 if backward_spacing != float('inf') else backward_spacing
            state = [forward_spacing, backward_spacing]
            if forward_spacing == float('inf') or backward_spacing == float('inf'):
                action, hold_time = 0.0, 0.0
                reward = None
            else:
                action, hold_time = self.infer(state)
                reward = self.calculate_reward(
                    forward_spacing, backward_spacing, locs)
            stop_bus_hold_time[(stop_id, route_id, bus_id)] = hold_time

            # record event for future use
            event = Event(
                time=snapshot.t,
                route_id=route_id,
                bus_id=bus_id,
                stop_id=stop_id,
                state=state,
                action=action,
                reward=reward
            )
            self._bus_stop_events[(route_id, bus_id)].append((stop_id, event))
            self._total_events.append(event)

            self.learn()
            self._learn_count += 1

        # if self._add_event_count > 100:
        return stop_bus_hold_time

    def infer(self, state: List[float]) -> Tuple[float, float]:
        state_ = torch.tensor(
            state, dtype=torch.float32).reshape(-1, self._state_size)
        with torch.no_grad():
            action = self._actor_net(state_)
            noise = np.random.normal(0, self._noise_level)
            action = (action + noise).clip(0, 1)
            action = float(action)
        hold_time = action * self._max_hold_time
        return action, hold_time

    def form_transition_tuple(self):
        ''' Form the transition tuple, including the graph data

        '''
        bus_stop_sar_graph: Dict[Tuple[str, str],
                                 Dict[str, SAR_Graph]] = defaultdict(dict)

        # for loop each bus's trajectory
        for (route_id, bus_id), stop_event_list in self._bus_stop_events.items():
            if len(stop_event_list) < 2:
                continue
            for (stop_id, event), (next_stop_id, next_event) in zip(stop_event_list[0:-1], stop_event_list[1:]):
                node_type, found_prev_stop_id = self._blueprint.get_previous_node(
                    route_id, next_stop_id)
                assert node_type != 'terminal', 'The previous node cannot be a terminal'
                assert found_prev_stop_id == stop_id, 'The previous stop is not the same as the current stop'
                visit_seq_stops = tuple(
                    self._blueprint.route_schema.route_details_by_id[route_id].visit_seq_stops)

                upstream_graph, downstream_graph = construct_graph(
                    bus_id, self._total_events, visit_seq_stops, stop_id, next_stop_id, event.time, next_event.time)

                if upstream_graph is None or downstream_graph is None:
                    continue

                sar_graph = SAR_Graph(state=event.state, action=event.action, reward=event.reward,
                                      upstream_graph=upstream_graph, downstream_graph=downstream_graph)
                bus_stop_sar_graph[(route_id, bus_id)][stop_id] = sar_graph

        # empty the bus_stop_events
        # self._bus_stop_events = defaultdict(list)
        self._bus_stop_events.clear()
        self._add_event_count = 0

        # connect bus_stop_sar_graph to form transition tuple
        for (route_id, bus_id), stop_sar_graph in bus_stop_sar_graph.items():
            visit_seq_stops = tuple(
                self._blueprint.route_schema.route_details_by_id[route_id].visit_seq_stops)
            if len(stop_sar_graph) < 2:
                continue

            for stop_id, sar_graph in stop_sar_graph.items():
                stop_id_idx = visit_seq_stops.index(stop_id)
                if stop_id_idx == len(visit_seq_stops) - 1:
                    continue
                next_stop_id = visit_seq_stops[stop_id_idx + 1]
                if next_stop_id not in stop_sar_graph:
                    continue
                next_sar_graph = stop_sar_graph[next_stop_id]

                sars_graph = SARS_Graph(
                    state=sar_graph.state,
                    action=sar_graph.action,
                    reward=next_sar_graph.reward,
                    next_state=next_sar_graph.state,
                    upstream_graph=sar_graph.upstream_graph,
                    downstream_graph=sar_graph.downstream_graph,
                    next_upstream_graph=next_sar_graph.upstream_graph,
                    next_downstream_graph=next_sar_graph.downstream_graph
                )
                if next_sar_graph.reward is None or sar_graph.state[0] == float('inf') or sar_graph.state[1] == float('inf'):
                    continue

                self._replay_buffer.append(sars_graph)

        bus_stop_sar_graph.clear()

    def calculate_reward(self, forward_spacing: float, backward_spacing: float, locs: List[float]):
        if forward_spacing == -1 or backward_spacing == -1:
            return None
        else:
            reward = -abs(forward_spacing - backward_spacing)
            return reward

    def learn(self):

        if self._learn_count % 250 != 0:
            return

        if len(self._replay_buffer) < self._batch_size:
            return

        print('learn....................', len(self._replay_buffer))
        self._actor_net.train()
        for _ in range(5):
            samples = random.sample(self._replay_buffer, self._batch_size)
            stats = []
            actis = []
            rewas = []
            next_stats = []
            for sample in samples:
                stats.append(sample.state)
                actis.append(sample.action)
                rewas.append(sample.reward)
                next_stats.append(sample.next_state)

            s = torch.tensor(
                stats, dtype=torch.float32).reshape(-1, self._state_size)
            # LongTensor for idx selection
            a = torch.tensor(actis, dtype=torch.float32)
            r = torch.tensor(rewas, dtype=torch.float32)
            n_s = torch.tensor(
                next_stats, dtype=torch.float32).reshape(-1, self._state_size)

            # update critic network
            self._critic_optim.zero_grad()
            # current estimate
            s_a = torch.concat((s, a.unsqueeze(dim=1)), dim=1)
            for param in self._critic_net.parameters():
                param.requires_grad = True
            Q = self._critic_net(s_a)

            # Bellman backup for Q function
            targe_imagi_a = self._target_actor_net(n_s)  # (batch_size, 1)
            s_targe_imagi_a = torch.concat((n_s, targe_imagi_a), dim=1)
            with torch.no_grad():
                q_polic_targe = self._target_critic_net(s_targe_imagi_a)
                # r is (batch_size, ), need to align with output from NN
                back_up = r.unsqueeze(1) + self._gamma * q_polic_targe
            # MSE loss against Bellman backup
            # Unfreeze Q-network so as to optimize it
            td = Q - back_up
            criti_loss = (td**2).mean()
            # update critic parameters
            criti_loss.backward()
            self._critic_optim.step()

            # update actor network
            self._actor_optim.zero_grad()
            imagi_a = self._actor_net(s)
            s_imagi_a = torch.concat((s, imagi_a), dim=1)
            # Freeze Q-network to save computational efforts
            for param in self._critic_net.parameters():
                param.requires_grad = False
            Q = self._critic_net(s_imagi_a)
            actor_loss = -Q.mean()
            actor_loss.backward()
            self._actor_optim.step()

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self._actor_net.parameters(), self._target_actor_net.parameters()):
                    p_targ.data.mul_(self._polya)
                    p_targ.data.add_((1 - self._polya) * p.data)
                for p, p_targ in zip(self._critic_net.parameters(), self._target_critic_net.parameters()):
                    p_targ.data.mul_(self._polya)
                    p_targ.data.add_((1 - self._polya) * p.data)

    def reset(self, episode: int):
        self.form_transition_tuple()
        self._total_events = []
        self._add_event_count = 0
        self._learn_count = 0
        self._noise_level = self._decay_rate ** episode * self._init_noise_level
        print('self._noise_level', self._noise_level)

    def save_net(self, path: str) -> None:
        torch.save(self._actor_net.state_dict(), path)

    def load_net(self, path):
        self._actor_net.load_state_dict(torch.load(path))
