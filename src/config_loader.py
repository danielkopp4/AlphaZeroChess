
import json
from os.path import join
from os import getenv

ROOT_PATH = getenv("CONFIG_PATH")
ROOT_PATH = "." if ROOT_PATH == None else ROOT_PATH

with open(join(ROOT_PATH, "config.json"), "r") as file:
    config: dict = json.load(file)
