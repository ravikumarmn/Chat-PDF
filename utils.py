import json 

def load_config():
    config = json.load(open("config/config.json", "r"))
    return config

