import yaml

def load_config(config_path):
    with open(config_path,"r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return configs