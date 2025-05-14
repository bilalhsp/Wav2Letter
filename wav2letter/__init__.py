import os
import yaml

config_dir = os.path.join(os.path.dirname(__file__), 'conf')

file_name = os.path.join(config_dir, 'config_rf.yaml')
with open(file_name, 'r') as f:
    manifest = yaml.load(f, Loader=yaml.FullLoader)