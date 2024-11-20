# EXACT: EXperimental Adaptive Control Testbed for Transit

This repository hosts a simulation testbed that serves as an environment for developing and benchmarking various dynamic bus holding strategies. It supports the research presented in the paper *"An Extensible Python Open-Source Simulation Platform for Developing and Benchmarking Bus Holding Strategies"* ([link to paper](https://ieeexplore.ieee.org/document/10720165)).

Users can customize any holding strategy within the platform, whether model-based or utilizing model-free Reinforcement Learning (RL), by creating a class that fulfills the basic requirements of the exposed API.

The platform currently provides two real-world datasets: Route No.3 in Chengdu, China, and a portion of the Guangzhou Bus Rapid Transit (BRT) corridor. The Chengdu Route 3 environment features high passenger demand and frequent dispatching, while the Guangzhou BRT environment offers a multi-line operation setting where queueing is commonly observed at stops. Users also have the flexibility to customize their own environment using their datasets, enabling the testing and evaluation of holding strategies across a wide range of scenarios.

For the control agent, we have currently furnished several control methods proposed in the literature:

- Model-based control strategies
  - `Forward Headway Control` proposed in Daganzo (2009).
  - `Simple Control` proposed in Xuan et al., (2011).
- Model-free RL control strategies:
  - Target-headway-based control proposed in Alesiani and Gkiotsalitis (2018).
  - Event-graph-based control proposed in Wang and Sun (2021).

# Environment setup

The following steps will guide you through the process of setting up the environment.

**Bulid virtual enviroment**

```bash
 conda create -n my_env python==3.9
 conda activate my_env
```

**Install Necessary Dependencies** use the following command：

```bash
 pip install -r requirements.txt
```

# Running Built-in Control Methods and Datasets

## **Configure the Algorithm and Simulation Environment**:

Specify the control algorithm and simulation environment in the `config.yaml` file.

## **Run the Simulation**:

After specifying the control agent and its associated parameters in the `config.yaml` file, run the main function through the terminal:

```bash
python main.py
```

# Customizing Your Own Environment

- Prepare your own dataset, including each route's station information, dispatch frequency, passenger demand (OD), and travel time between stations.

- Inherit the class **Network** and implement the provided methods to create a bus network with the route(s)'s information specified.

- Implement the `Components_Factory` protocol and then add it to the `Builder` class. This factory will create the related components used in the simulation, including terminals, stops, links, and a holder.

# Customizing Your Own RL Algorithm

Inherit the `RLAgent` class from the `rl_agent` module and implement your specific algorithm in the `calculate_hold_time`. This function will provide you the `snapshot` that record all potentially required states, including bus ID, route ID, number of passengers (pax_num), location relative to the terminal (loc_relative_to_terminal), and etc.

# References
[1] Daganzo, C. F., 2009. A headway-based approach to eliminate bus bunching: Systematic analysis and comparisons. Transportation Research Part B: Methodological 43 (10), 913–921.

[2] Xuan, Y., Argote, J., Daganzo, C. F., 2011. Dynamic bus holding strategies for schedule reliability: Optimal linear control and performance analysis. Transportation Research Part B: Methodological 45 (10), 1831–1845.

[3] Alesiani, F., Gkiotsalitis, K., 2018. Reinforcement learning-based bus holding for high-frequency services. In: 2018 21st International Conference on Intelligent Transportation Systems (ITSC). IEEE, pp. 3162–3168.

[4] Wang, J., Sun, L., 2021. Reducing bus bunching with asynchronous multi-agent reinforcement learning. arXiv preprint arXiv:2105.00376.

# For any questions, feel free to contact us via

[1221201Z5005@smail.swufe.edu.cn](mailto:1221201Z5005@smail.swufe.edu.cn) or [shenminyu@swufe.edu.cn](mailto:shenminyu@swufe.edu.cn)

