from runner import run
from config import build_simulation_elements
import pprint

blueprint, agent, run_config, record_config = build_simulation_elements()
name_metric, trip_times = run(blueprint, agent, run_config, record_config)
