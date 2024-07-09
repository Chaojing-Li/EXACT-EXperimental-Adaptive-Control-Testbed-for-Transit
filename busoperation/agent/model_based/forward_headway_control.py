from typing import Dict, Any, Tuple
from typing_extensions import TypedDict
from copy import deepcopy

from setup.blueprint import Blueprint
from simulator.virtual_bus import VirtualBus
from simulator.snapshot import Snapshot
from simulator.simulator import Simulator

from ..single_line_agent import AgentByLine


class ForwardHeadwayControl(AgentByLine):
    ''' 

    Attributes:
        _slack: specify the slack time
        _alpha: specify the control parameter alpha
        _base_type: specify when to perform the control:
            when the bus arrives to the stop (`arrival`) or when the bus is ready to depart the stop ('rtd')

    '''

    def __init__(self, agent_config: Dict[str, Any], blueprint: Blueprint, run_config: Dict) -> None:
        super().__init__(agent_config, blueprint)
        self._slack = agent_config['slack']
        self._alpha = agent_config['alpha']
        self._is_nonlinear = agent_config['is_nonlinear']
        self._base_type = agent_config['base_type']

        self._blueprint = blueprint
        self._run_config = run_config
        self._episode_num_for_stabilize_average_hold = agent_config[
            'episode_num_for_stabilize_average_hold']
        self._episode_duration_for_stabilize_average_hold = agent_config[
            'episode_duration_for_stabilize_average_hold']
        self._generate_virtual_bus()

    def calculate_hold_time(self, snapshot: Snapshot) -> Dict[Tuple[str, str, str], float]:
        ''' Implement the nonlinear control algorithm.

        Args:
            snapshot: Snapshot

        Returns:
            stop_bus_hold_time: a dictionary {(stop_id, route_id, bus_id) -> hold_time}

        '''
        stop_bus_hold_time = {}
        action_buses = snapshot.holder_snapshot.action_buses
        if len(action_buses) == 0:
            return stop_bus_hold_time

        for (stop_id, route_id, bus_id) in action_buses:
            if not snapshot.bus_snapshots[(route_id, bus_id)].is_need_to_hold:
                stop_bus_hold_time[(stop_id, route_id, bus_id)] = 0
                continue

            _, forward_spacing, _, backward_spacing = self.extract_local_info_from_snapshot(
                bus_id, snapshot, ['spacing'])

            stop_boarding_rate = self._blueprint.route_schema.route_details_by_id[
                route_id].boarding_rate
            arrival_rate = self._route_stop_arrival_rate[route_id][stop_id]
            beta = arrival_rate / stop_boarding_rate[stop_id]
            H = self._route_schedule[route_id]

            last_rtd_time = snapshot.get_last_rtd_time(route_id, stop_id)
            current_time = snapshot.t
            h = current_time - last_rtd_time

            # get the current bus's `epsilon_arrival` and `epsilon_rtd` at the current stop
            epsilon_arrival_curr_stop, epsilon_rtd_curr_stop = snapshot.get_bus_epsilon(
                route_id, bus_id, stop_id)

            # get the last bus's epsilon_arrival and epsilon_rtd at the current stop
            last_bus_epsilon_arrival_curr_stop, last_bus_epsilon_rtd_curr_stop = snapshot.get_stop_epsilon(
                route_id, stop_id, bus_id)

            # verify if the values are calculated correctly
            numerical_diff = abs(
                (h-H) - (epsilon_rtd_curr_stop - last_bus_epsilon_rtd_curr_stop))
            if numerical_diff > 1:
                print('A mismatch between h-H and epsilon difference...')

            hold_time = 0
            if self._base_type == 'arrival':
                # hold_time = (self._alpha + beta) * (H-h)
                # hold_time += self._slack
                pass
            elif self._base_type == 'rtd':
                hold_time = self._alpha * (H-h)
                hold_time += self._slack

            if forward_spacing == float('inf') or backward_spacing == float('inf'):
                hold_time = 0

            if self._is_nonlinear:
                hold_time = max(0, hold_time)
            stop_bus_hold_time[(stop_id, route_id, bus_id)] = hold_time

        return stop_bus_hold_time

    def _generate_virtual_bus(self):
        ''' Generate the virtual bus.

        For nonlinear version, the average holding time at each stop need to be dynamically updated
        by running the simulation until convergence. The average holding time is initialized to be the slack.
        The episode number and duration for stabilizing the average holding time are specified in the configuration.


        '''
        # the virtual bus's average holding time is initialized to be the slack
        self._virtual_bus = VirtualBus(self._blueprint)
        self._virtual_bus.initialize_with_perfect_schedule(
            self._route_stop_arrival_rate, self._slack)

        if self._episode_num_for_stabilize_average_hold == 0:
            print('Do not stabilize the average hold time for the virtual bus ......')
            return

        route_stop_average_hold_time: Dict[str, Dict[str, float]] = {}
        for _ in range(self._episode_num_for_stabilize_average_hold):
            simulator = Simulator(self._blueprint, self, self._run_config)
            stop_bus_hold_action: Dict[Tuple[str, str, str], float] = {}
            for t in range(self._episode_duration_for_stabilize_average_hold):
                snapshot = simulator.step(t, stop_bus_hold_action)
                stop_bus_hold_action = self.calculate_hold_time(snapshot)
                snapshot.record_holding_time(stop_bus_hold_action)

            route_stop_average_hold_time = simulator.get_stop_average_hold_time()
            self._virtual_bus.update_trajectory(route_stop_average_hold_time)

    def reset(self, episode: int) -> None:
        ''' Reset the agent for the next episode
        '''
        pass
