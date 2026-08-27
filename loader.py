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
    def __init__(self, argparser):
        self.models = {}
        self.argparser = argparser
        self.args = argparser.parser.parse_args()
        self.configs = json.load((ROOT / Path("configs.json")).open("r", encoding='utf-8'))
        
        models_path = Path(self.configs["models_paths"])
        json_paths = models_path.rglob("*.json")

        self.profiles_json = json.load((ROOT / Path("profiles.json")).open("r", encoding='utf-8')) 
        self.llamaui_json = json.load((ROOT / Path("llamaui.json")).open("r", encoding='utf-8'))

        for json_path in json_paths:
            model_json = json.load(json_path.open("r", encoding='utf-8'))
            if {"name", "files", "alias"} <= model_json.keys():
                self.models[model_json["alias"]] = Model(model_json, json_path, json_path.parent, self.profiles_json, self.llamaui_json)
            

    def start(self, model:str, llamaargs: list | None = None, b: bool = False, i: bool = False):
        """ Start the server. Looks for the first index of the flags list to check if it's a profile or llama flag """        
        profile_arg = llamaargs[0]  # Collect the first argument, the possible profile
        if profile_arg in self.profiles_json.keys():  # Check if it's in the list of profiles
            llamaargs.pop(0)  # If it's in, then pop the first item of the list to prevent further errors
            profile_json = self.profiles_json[profile_arg].copy()  # Create a copy of the profile
            system = profile_json.pop("systemMessage")  # Pop the system message
            
            flags_dict = self.argparser.args_to_dict(llamaargs)  # Dictionary with the llama flags

            model = self.models[model].copy()  # Define the correct model's arguments
            model.arguments.update(profile_json)  # Update with the profile chosen
            model.arguments.update(flags_dict)  # Update the flags

            command = model.convert(model.arguments)  # Convert the arguments to a Popen command


            if b or i:  # Check if the browser or incognito flag is set
                self._open_browser(arguments, i)  # If it is, then open the browser. NOTE: The llama-ui WAITS for the model to load, this is not a BUG!

            # Just start the process with that command
            process = subprocess.Popen(command)

            try:
                process.wait()  # Waits for the process
            except KeyboardInterrupt:  # If the user uses CTRL + C
                print("\nClosing the server...")
                process.terminate()
                process.wait()
        

    def list(self):
        for alias, modelo in self.models.items():
            print(f"alias: {alias}, name: {modelo.name}, path: {modelo.parent}")


    def edit(self, objeto: str):
        path = ROOT / Path(f"{objeto}.json") if objeto in ("configs", "profiles", "llamaui") else self.models[objeto].path
        subprocess.Popen(["code.cmd", path])


    def run(self):
        match self.args.command:
            case "list":
                self.list()
            case "edit":
                self.edit(self.args.object)
            case "start":
                self.start(self.args.model, self.args.llamaargs, self.args.b, self.args.i)
            case _:
                self.argparser.parser.print_help()



    def _open_browser(self, arguments: dict, incognito: bool = False):        
        command = [
            Path(self.configs["browser_path"]),
            "--start-maximized",
            f"http://{arguments["--host"]}:{arguments["--port"]}"
            ]
        if incognito:
            command.append("--incognito")
            
        subprocess.Popen(command)
        


class Argparser:
    """ Class responsable """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.subparser = self.parser.add_subparsers(dest="command")

        self.list_parser = self.subparser.add_parser("list")

        self.edit_parser = self.subparser.add_parser("edit")
        self.edit_parser.add_argument("object", help="The .json object. You can edit any model's json or chose to edit configs, profiles, llamaui")

        self.start_parser = self.subparser.add_parser("start")
        self.start_parser.add_argument("model", help="The model's alias.")
        self.start_parser.add_argument("-b", action='store_true', help="Start the browser.")
        self.start_parser.add_argument("-i", action='store_true', help="Start the browser in incognito.")
        self.start_parser.add_argument("llamaargs", nargs=argparse.REMAINDER, help="Aditional llama.cpp flags.")


    def args_to_dict(self, args: list[str]) -> dict[str, str]:
            """ Given an llamaargs list, turn it into a dict """
            def is_flag(value: str) -> bool:
                """ Auxiliary parser, just to check if the argument is a flag or not """
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



if __name__ == '__main__':
    argparser = Argparser()    
    loader = Loader(argparser)

    loader.run()
    

