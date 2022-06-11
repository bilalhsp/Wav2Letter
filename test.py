import yaml
import os
path = os.path.join(os.getcwd() ,'conf/puzzlelib_model.yaml')
with open(path, 'r') as f:
    manifest = yaml.load(f, Loader=yaml.FullLoader)


print("Hello, I am here, the path to load from is: ")
print(path)
print(manifest.keys())
print(manifest['results_dir'])
