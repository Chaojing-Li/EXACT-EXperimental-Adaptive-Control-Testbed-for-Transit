from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np
import wandb

from agent.agent import Agent
from agent.rl.rl_agent import RLAgent
from setup.blueprint import Blueprint
from simulator.simulator import Simulator
from simulator.trajectory import plot_time_space_diagram


def run(blueprint: Blueprint, agent: Agent, run_config: Dict, record_config: Dict) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    ''' Run the simulation for multiple episodes and return the metrics

    Given a `blueprint` and an `agent`, run the simulation according to the `run_config` for one or multiple times and return the metrics. 
    If `is_record_wandb` in `config.yaml` is specified, the metrics of each episode will be recorded in `wandb`.

    Args:
        blueprint: blueprint that provide network and route information as a whole
        agent: specific Agent object
        run_config: configuration for running the simulation
        record_config: configuration for recording in wandb, an empty dict if not need recording

    Returns:
        name_metric_value: a dict mapping the name of the metric to its value, averaged across all episodes
        route_trip_times: a dict mapping the route id to a list of trip times that all buses experience in all episodes

    '''
    is_record = True if len(record_config) > 0 else False
    if is_record:
        wandb_project_name = record_config['wandb_config']['wandb_project_name']
        wandb.init(project=wandb_project_name, config=record_config)

    name_episode_metrics: Dict[str, List[float]] = defaultdict(list)
    route_trip_times: Dict[str, List[float]] = defaultdict(list)

    for epsisode in range(run_config['episode_num']):
        # at the beginning of each episode, we reset the simulator
        simulator = Simulator(blueprint, agent, run_config)

        # stop_bus_hold_action: {(stop_id, route_id, bus_id) -> specified holding time}
        stop_bus_hold_action: Dict[Tuple[str, str, str], float] = {}

        # main opeartion loop for each episode
        for t in range(run_config['episode_duration']):
            snapshot = simulator.step(t, stop_bus_hold_action)
            stop_bus_hold_action = agent.calculate_hold_time(snapshot)
            snapshot.record_holding_time(stop_bus_hold_action)

        # get the metrics for each episode and store them
        metrics, route_dispatch_time_trip_time = simulator.get_metrics()
        for name, metric in metrics.items():
            name_episode_metrics[name].append(metric)

        # get the route trip times
        for route, dispatch_time_trip_time in route_dispatch_time_trip_time.items():
            for dispatch_time, trip_time in dispatch_time_trip_time.items():
                route_trip_times[route].append(trip_time)
        route_trip_times = dict(route_trip_times)

        print(f'---------- episode {epsisode} ------------')
        print(f'metrics is {metrics}')
        agent.reset(epsisode)
        if epsisode == run_config['episode_num'] - 1:
            # plot_time_space_diagram(simulator.total_buses)
            # save the model if the agent is an RL agent and it is training
            if isinstance(agent, RLAgent) and agent.is_train:
                agent.save_net(path='actor_net.pth')

        if is_record:
            wandb.log(metrics)
    if is_record:
        wandb.finish()

    # return the averaged metrics across all episodes, and the route trip times
    name_metric_value = {}
    for name, episode_metrics in name_episode_metrics.items():
        metric_mean = np.mean(np.array(episode_metrics))
        name_metric_value[name] = metric_mean
    return name_metric_value, route_trip_times
