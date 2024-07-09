from typing import List, Dict, Tuple, Literal
from collections import defaultdict

from agent.agent import Agent
from setup.blueprint import Blueprint

from .builder import Builder
from .bus import Bus
from .holder import Holder
from .link import Link
from .mediator import Mediator
from .pax import PaxGenerator, Pax
from .snapshot import Snapshot
from .stop import Stop
from .terminal import Terminal
from .tracer import Tracer


class Simulator:
    ''' The simulator that emulates the operation of a bus system.

    Properties:
        total_buses: all the buses that have been dispatched from terminals

    Methods:
        step(self, t: int, stop_bus_hold_times: Dict[Tuple[str, str, str], float]) -> Snapshot
        take_snapshot(self, t: int) -> Snapshot
        get_metrics(self) -> Tuple[Dict[str, float], Dict[str, Dict[int, int]]]
        get_stop_average_hold_time(self) -> Dict[str, Dict[str, float]]

    '''

    def __init__(self, blueprint: Blueprint, agent: Agent, run_config: Dict) -> None:
        self._blueprint: Blueprint = blueprint
        self._agent: Agent = agent
        self._run_config: Dict = run_config

        # A builder is used to create all the components in the simulation
        self._builder: Builder = Builder(blueprint, agent)
        # The metric names that need to be calculated
        self._metric_names: List[Literal['headway_std', 'schedule_deviation', 'pax_in_vehicle_wait_time',
                                         'pax_out_vehicle_wait_time', 'hold_time', 'queueing_delay']] = run_config['metric_names']

        hold_period = (run_config['hold_start_time'],
                       run_config['hold_end_time'])
        has_schedule = run_config['has_schedule']

        # A virtual bus is used to specify the initial condition of the dynamics, i.e., passenger arrival start time at each stop
        # If the `agent`` has created a virtual bus (by repeatedly running simulation in agent's init method and taking the convergent hold time), use it;
        # This is the typical case for the nonlinear control where slack is used as a control paratemer and max{0, x} is used to ensure nonnegative hold time
        # Otherwise, create a virtual bus with perfect schedule (without repeated simulation)
        if hasattr(agent, 'virtual_bus'):
            self._virtual_bus = agent.virtual_bus
        else:
            self._virtual_bus = self._builder.create_virtual_bus()

        # Pax generator for generating passengers at all stops
        self._pax_generator: PaxGenerator = self._builder.create_pax_generator(
            self._virtual_bus)

        # Terminals that dispatch and recycle buses
        self._terminals: Dict[str, Terminal] = self._builder.create_terminals(
            self._virtual_bus, hold_period)

        # Links that buses run on
        self._links: Dict[str, Link] = self._builder.create_links()

        # Stops that buses stop at to pick up and drop off passengers
        self._stops: Dict[str, Stop] = self._builder.create_stops(
            self._virtual_bus, has_schedule)

        # Holder that holds buses after they finish their operation at a stop
        self._holder: Holder = Holder(
            self._agent, self._virtual_bus, has_schedule)

        # A mediator is used to transfer buses between components
        # i.e., between terminals, links, stops, and holder
        self._mediator: Mediator = Mediator(
            blueprint, self._terminals, self._links, self._stops, self._holder)

        # A tracer is used to record the status of the simulation
        self._tracer: Tracer = Tracer()

        # Maintain a list of all the buses that have been dispatched from terminals
        # used for time-space diagram visualization in the end
        self._total_buses: List[Bus] = []

        # used for recording the passengers that leave the system
        self._left_paxs: List[Pax] = []

        # self._blueprint.network.visualize()

    @property
    def total_buses(self) -> List[Bus]:
        ''' Get all the buses that have been dispatched from terminals.

        '''
        return self._total_buses

    def step(self, t: int, stop_bus_hold_time: Dict[Tuple[str, str, str], float]) -> Snapshot:
        '''Accept holding actions and move buses one step forward

        Args:
            t: current time
            stop_bus_hold_times: {(stop_id, route_id, bus_id): specified holding time}

        Returns:
            Snapshot: a snapshot of current time t
        '''

        # 0. dispatch buses from terminal to their first links
        for terminal_id, terminal in self._terminals.items():
            dispatching_buses = terminal.dispatch(t)
            self._mediator.transfer(
                dispatching_buses, 'terminal', terminal_id, t)
            # record all the dispatched buses for future visualization
            for bus in dispatching_buses:
                self._total_buses.append(bus)

        # 1. passengers arrive at stops
        stop_paxs = self._pax_generator.generate(t)
        for stop_id, paxs in stop_paxs.items():
            self._stops[stop_id].pax_arrive(paxs)

        # 2. link operation
        for link_id, link in self._links.items():
            leaving_link_buses = link.forward(t)
            self._mediator.transfer(leaving_link_buses, 'link', link_id, t)

        # 3. stop operation
        for stop_id, stop in self._stops.items():
            leaving_stop_buses, leaving_paxs = stop.operation(t)
            self._mediator.transfer(leaving_stop_buses, 'stop', stop_id, t)

            for pax in leaving_paxs:
                self._left_paxs.append(pax)

        # 4. holding operation
        self._holder.set_hold_action(stop_bus_hold_time)
        stop_held_buses = self._holder.operation(t)
        # transfer buses that finish holding to the next link
        for stop_id, held_buses in stop_held_buses.items():
            self._mediator.transfer(held_buses, 'holder', stop_id, t)

        # 5. count in-vehicle delay for passengers on the bus
        # the out-vehicle delay is counted within bus stop's operation function
        for bus in self._total_buses:
            if bus.status != 'finished':
                bus.accumulate_in_vehicle_delay()

        snapshot = self.take_snapshot(t)
        return snapshot

    def take_snapshot(self, t: int) -> Snapshot:
        ''' Take a snapshot of the whole current state of the simulation.

        '''
        snapshot = self._tracer.take_snapshot(
            t, self._links, self._stops, self._holder)
        return snapshot

    def get_metrics(self) -> Tuple[Dict[str, float], Dict[str, Dict[int, int]]]:
        ''' Get the metrics of the simulation.

        Generally called after one episode of simulation finished.

        Returns:
            metrics: a dictionary of metrics
            route_dispatch_time_trip_time: a dictionary {route_id -> {dispatch_time -> trip_time}}
                dispatch_time is the time when the bus is dispatched from the terminal
                trip_time is the duration of the trip from the terminal to the ending terminal

        '''
        # stats all the stops (excluding the starting and ending terminal stops)
        route_stats_stop_ids: Dict[str, List[str]] = defaultdict(list)
        for route_id, route in self._blueprint.route_schema.route_details_by_id.items():
            route_stats_stop_ids[route_id].extend(route.visit_seq_stops)

        metrics = self._tracer.get_metric(
            route_stats_stop_ids, self._total_buses, self._left_paxs, metric_names=self._metric_names)

        # stats the trip time
        route_dispatch_time_trip_time: Dict[str, Dict[int, int]] = {}
        for route_id, route in self._blueprint.route_schema.route_details_by_id.items():
            dispatch_time_trip_time = {}
            for bus in self._total_buses:
                if bus.route_id == route_id:
                    if bus.bus_log.end_time is not None:
                        assert bus.bus_log.dispatch_time is not None
                        trip_time = bus.bus_log.end_time - bus.bus_log.dispatch_time
                        dispatch_time_trip_time[bus.bus_log.dispatch_time] = trip_time
            route_dispatch_time_trip_time[route_id] = dispatch_time_trip_time
        return metrics, route_dispatch_time_trip_time

    def get_stop_average_hold_time(self) -> Dict[str, Dict[str, float]]:
        ''' Get the average holding time at each stop for each route.

        Typically called after one episode of simulation finished.

        '''
        route_stop_average_hold_time = self._tracer.get_stop_average_hold_time()
        return route_stop_average_hold_time
