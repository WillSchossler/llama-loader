import json
import subprocess
from pathlib import Path


class Model:
    """ Stores the model data """
    def __init__(self, model: dict, parent: Path, profiles_json: dict, llamaui_json: dict):
        self.parent = parent
        self.profiles_json = profiles_json
        self.llamaui_json = llamaui_json.copy()

        self.name = model["name"]
        self.alias = model["alias"]
        self.profile = model["profile"]
        self.parameters = model["parameters"]

        
        self.profile_json = self.profiles_json[self.profile].copy()
        self.system = self.profile_json.pop("systemMessage", None)
        if self.system is not None:
            self.llamaui_json["systemMessage"] = self.system
        

        self.files = {}        
        for (file, path) in model["files"].items():
            if path is not None:
                self.files[file] = parent / path

        self.parser = {}  
        self.parser.update(self.profiles_json["defaults"])  
        self.parser.update(self.profile_json)  
        self.parser.update(self.parameters)  
        self.parser.update(self.files)         
        self.parser["--ui-config"] = json.dumps(self.llamaui_json, ensure_ascii=False)

        self.command = ["llama-server"]
        for parameter, value in self.parser.items():
            self.command.append(parameter)
            if value != "":
                self.command.append(str(value))



class Loader:
    def __init__(self):
        self.models = {}
        self.process = None
        self.active_model = None            
        self.configs = json.load(Path("config.json").open("r", encoding='utf-8'))


        self.__load_models()


    def start(self, model: str):
        if self.is_running:
            raise RuntimeError("Já existe um servidor em execução.")
        else:
            self.active_model = self.models[model]
            self.process = subprocess.Popen(
                self.active_model.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                )


    @property
    def is_running(self):
        return self.process is not None and self.process.poll() is None


    def stop(self):
        if self.is_running:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait()
                
                self.process = None
                self.active_model = None
    

    def open_browser(self, model: Model, incognito: bool = False):
        """ Open the browser for the given model """
        browser_path = Path(self.configs["browser_path"]) 
        
        command = [
            str(browser_path),
            "--start-maximized",
            f"http://{model.parser["--host"]}:{model.parser["--port"]}"
            ]
        if incognito:
            command.append("--incognito")
            
        subprocess.Popen(command)


    def __load_models(self):
        """ Loads all the models into "models" dict """
        models_path = Path(self.configs["models_paths"])
        json_paths = models_path.rglob("*.json")

        profiles_json = json.load(Path(self.configs["profiles_path"]).open("r", encoding='utf-8')) 
        llamaui_json = json.load(Path(self.configs["llamaui_path"]).open("r", encoding='utf-8'))

        for json_path in json_paths:
            model_json = json.load(json_path.open("r", encoding='utf-8'))
            if {"name", "files", "alias"} <= model_json.keys():
                self.models[model_json["alias"]] = Model(model_json, json_path.parent, profiles_json, llamaui_json)




loader = Loader()
loader.start("gemma")


for line in loader.process.stderr:
    if "127.0.0.1:9931" in line:
        print(line, end="")