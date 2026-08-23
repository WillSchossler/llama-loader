import json
import subprocess
from pathlib import Path


models = {}
configs = json.load(Path("config.json").open("r", encoding='utf-8'))

browser_path = Path(configs["browser_path"])
models_path = Path(configs["models_paths"])
json_paths = models_path.rglob("*.json")

profiles_json = json.load(Path(configs["profiles_path"]).open("r", encoding='utf-8')) 
llamaui_json = json.load(Path(configs["llamaui_path"]).open("r", encoding='utf-8'))


class Model:
    """ Stores the model data """
    def __init__(self, model: dict, parent: Path, profiles_json: dict, llamaui_json: dict):
        self.parent = parent
        self.profiles_json = profiles_json.copy()
        self.llamaui_json = llamaui_json.copy()

        self.name = model["name"]
        self.alias = model["alias"]
        self.profile = model["profile"]
        self.parameters = model["parameters"]

        self.profile_json = self.profiles_json[self.profile]
        self.system = self.profile_json.pop("systemMessage", None)
        self.llamaui_json["systemMessage"] = self.system

        self.files = {}        
        for (file, path) in model["files"].items():
            if path is not None:
                self.files[file] = parent / path


        self.parser = {}  
        self.parser.update(self.profiles_json["defaults"])  
        self.parser.update(self.profiles_json[self.profile])  
        self.parser.update(self.parameters)  
        self.parser.update(self.files)         
        self.parser["--ui-config"] = json.dumps(self.llamaui_json, ensure_ascii=False)


    def start(self):
        command = ["llama-server"]

        for parameter, value in self.parser.items():
            command.append(parameter)

            if value != "":
                command.append(str(value))

        subprocess.run(command)


for json_path in json_paths:
    model_json = json.load(json_path.open("r", encoding='utf-8'))
    if {"name", "files", "alias"} <= model_json.keys():
        models[model_json["alias"]] = Model(model_json, json_path.parent, profiles_json, llamaui_json)
        

models["gemma"].start()