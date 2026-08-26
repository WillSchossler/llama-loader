import json
import argparse
import subprocess
from pathlib import Path
ROOT = Path(__file__).parent

class Model:
    """ Stores the model data """
    def __init__(self, model: dict, path: Path, parent: Path, profiles_json: dict, llamaui_json: dict):
        self.path = path
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
                self.files[file] = str(parent / path)


        self.arguments = {}
        for argument in [
            self.profiles_json["defaults"],
            self.profile_json,
            self.parameters,
            self.files
            ]:
            
            self.arguments.update(argument)         
        self.arguments["--ui-config"] = json.dumps(self.llamaui_json, ensure_ascii=False)



    def convert(self, parameters: dict) -> list:
        """ Convert a dictionary into a llama-server list ready to run """
        command = ["llama-server"]
        for parameter, value in parameters.items():
            command.append(parameter)
            if value != "":
                command.append(str(value))
        
        return command



class Loader:
    def __init__(self):
        self.models = {}
        self.configs = json.load((ROOT / Path("configs.json")).open("r", encoding='utf-8'))
        
        models_path = Path(self.configs["models_paths"])
        json_paths = models_path.rglob("*.json")

        profiles_json = json.load((ROOT / Path(self.configs["profiles_path"])).open("r", encoding='utf-8')) 
        llamaui_json = json.load((ROOT / Path(self.configs["llamaui_path"])).open("r", encoding='utf-8'))

        for json_path in json_paths:
            model_json = json.load(json_path.open("r", encoding='utf-8'))
            if {"name", "files", "alias"} <= model_json.keys():
                self.models[model_json["alias"]] = Model(model_json, json_path, json_path.parent, profiles_json, llamaui_json)
            

    def start(self, model:str, llamaargs: list | None = None, b: bool = False, i: bool = False):

        def args_to_dict(args: list[str]) -> dict[str, str]:
            
            
            def is_flag(value: str) -> bool:
                if not value.startswith("-"):
                    return False

                try:
                    float(value)
                    return False
                except ValueError:
                    return True

            result = {}
            i = 0

            while i < len(args):
                key = args[i]

                if i + 1 >= len(args):
                    result[key] = ""
                    break

                next_value = args[i + 1]

                if is_flag(next_value):
                    result[key] = ""
                    i += 1
                else:
                    result[key] = next_value
                    i += 2

            return result


        model = self.models[model]

        arguments = model.arguments.copy()
        arguments.update(args_to_dict(llamaargs))

        command = model.convert(arguments)                

        if b:
            self._open_browser(model, i)

        process = subprocess.Popen(command)

        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nEncerrando servidor...")
            process.terminate()
            process.wait()
        

    def list(self):
        for alias, modelo in self.models.items():
            print(f"alias: {alias}, name: {modelo.name}, path: {modelo.parent}")


    def edit(self, objeto: str):
        path = ROOT / Path(f"{objeto}.json") if objeto in ("configs", "profiles", "llamaui") else self.models[objeto].path
        subprocess.Popen(["code.cmd", path])


    def run(self, arguments):
        match arguments.command:
            case "list":
                self.list()
            case "edit":
                self.edit(arguments.object)
            case "start":
                self.start(arguments.model, arguments.llamaargs, arguments.b, arguments.i)



    def _open_browser(self, model: Model, incognito: bool = False):        
        command = [
            Path(self.configs["browser_path"]),
            "--start-maximized",
            f"http://{model.arguments["--host"]}:{model.arguments["--port"]}"
            ]
        if incognito:
            command.append("--incognito")
            
        subprocess.Popen(command)

        



class Argparser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.subparser = self.parser.add_subparsers(dest="command", help="""
        list: List all the available models.
        edit <model/configs/profiles/llamaui>: Opens the editor to manually edit any model, or the configs/profiles/llamaui json.
        start <model> -b, -bi, flags: Start the given model. You can pass ANY llama.cpp valid flags you want. Use -b to open the browser, or -bi to open incognito.
        Exemple: start qwen -bi --cache-type-k q8_0
        """)

        self.list_parser = self.subparser.add_parser("list")

        self.edit_parser = self.subparser.add_parser("edit")
        self.edit_parser.add_argument("object", help="The .json object. You can edit any model's json or chose to edit configs, profiles, llamaui")

        self.start_parser = self.subparser.add_parser("start")
        self.start_parser.add_argument("model", help="The model's alias.")
        self.start_parser.add_argument("-b", action='store_true', help="Start the browser.")
        self.start_parser.add_argument("-i", action='store_true', help="Start the browser in incognito.")
        self.start_parser.add_argument("llamaargs", nargs=argparse.REMAINDER, help="Aditional llama.cpp flags.")


if __name__ == '__main__':
    argparser = Argparser()
    args = argparser.parser.parse_args()
    
    loader = Loader()
    loader.run(args)
    

