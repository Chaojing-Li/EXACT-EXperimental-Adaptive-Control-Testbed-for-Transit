# EXACT: EXperimental Adaptive Control Testbed for Transit

This repository hosts a simulation testbed that can be used as an environment for developing and comparing different dynamic holding strategies. Any holding strategy, whether model-based or model-free RL (Reinforcement Learning), can be customized by users within our platform, provided they create a class that fulfills the basic requirements of the exposed API.

The platform currently provides two real-world datasets, a Route No.3 in Chengdu, China and a portion of the Guangzhou Bus Rapid Transit (BRT) corridor in China. The Chengdu route 3 environment features featuring high passenger demand and high dispatch frequency. The Guangzhou BRT environment offers a multi-line operation setting where queueing is frequently observed at stops. You can also customize your own environment using your dataset; see the usage instructions for details.

For the control agent, we have currently furnished several control methods proposed in the literature:

- Model-based control strategies
  - `Forward Headway Control` proposed in Daganzo (2009).
  - `Simple Control` proposed in Xuan et al., (2011).
- Model-free RL control strategies:
  - Target-headway-based control proposed in Alesiani and Gkiotsalitis (2018).
  - Event-graph-based control proposed in Wang and Sun (2021).

# Environment setup

The follwing steps guide you thorough how to build up the environment for running the `EXACT`.

**Bulid virtual enviroment**

```bash
 conda create -n my_env python==3.9
 conda activate my_env
```

**Install Necessary Dependencies** use the following command：

```bash
 pip install -r requirements.txt
```

# Running built-in control methods and datasets

## 1. **Configure the Algorithm and Simulation Environment**:

Specify the control algorithm and simulation environment in the `config.yaml` file.

## 2. **Run the Simulation**:

After specifying the control agent and its associated parameters in the `config.yaml` file, run the main function through the terminal:

```bash
python main.py
```

# Customizing Your Own Environment

1. Prepare your own dataset, including each route's station information, dispatch frequency, passenger demand (OD), and travel time between stations.

2. Inherit the class **Network** and implement the provided methods to create a bus network with the route(s)'s information specified.

3. Implement the `Components_Factory` protocol and then add it to the `Builder` class. This factory will create the related components used in the simulation, including terminals, stops, links, and a holder.

# Customizing Your Own RL Algorithm

Inherit the `RLAgent` class from the `rl_agent` module and implement your specific algorithm in the `calculate_hold_time`. This function will provide you the `snapshot` that record all potentially required states, including bus ID, route ID, number of passengers (pax_num), location relative to the terminal (loc_relative_to_terminal), and etc.

# For any questions, feel free to contact us via

[1221201Z5005@smail.swufe.edu.cn](mailto:1221201Z5005@smail.swufe.edu.cn) or [shenminyu@swufe.edu.cn](mailto:shenminyu@swufe.edu.cn)
