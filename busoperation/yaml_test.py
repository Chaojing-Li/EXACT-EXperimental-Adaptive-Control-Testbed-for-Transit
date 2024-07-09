import yaml
import pprint

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

pprint.pprint(config)
