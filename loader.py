import json
import argparse
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent



class Argparser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="TODO", prog="llama-loader")
        self.subparser = self.parser.add_subparsers(dest="command", required=True, help="TODO")


        self.list_parser = self.subparser.add_parser("list")
        list_group = self.list_parser.add_mutually_exclusive_group()
        list_group.add_argument("--models", "-m", action="store_true", help="TODO")
        list_group.add_argument("--profiles", "-p", action="store_true", help="TODO")


        self.edit_parser = self.subparser.add_parser("edit")
        self.edit_parser.add_argument("file", help="TODO")


        self.show_parser = self.subparser.add_parser("show")
        self.show_parser.add_argument("object", help="TODO")
        self.show_parser.add_argument("profile", nargs="?", help="TODO")


        self.start_parser = self.subparser.add_parser("start")
        self.start_parser.add_argument("model", help="TODO")
        start_group = self.start_parser.add_mutually_exclusive_group()
        start_group.add_argument("-b", action='store_true', help="TODO")
        start_group.add_argument("-i", action='store_true', help="TODO")
        start_group.add_argument("-a", action='store_true', help="TODO")
        self.start_parser.add_argument("llamaargs", nargs=argparse.REMAINDER, help="TODO")


    def parse(self) -> argparse.Namespace:
        return self.parser.parse_args()


    @staticmethod
    def args_to_dict(args: list[str]) -> dict[str, str]:
            """ Converts a list of llama.cpp arguments to a valid dictionary with keys and values. """
            def is_flag(value: str) -> bool:
                """ Auxiliary parser used to check if the argument is a flag or not """
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



class Model:
    """ Stores model data """
    def __init__(self, model: dict, path: Path, parent: Path, profiles_json: dict):
        self.path = path
        self.parent = parent
        self.name = model["name"]
        self.alias = model["alias"]
        self.profile = model["profile"]
        self.parameters = model["parameters"]
        

        self.files = {}        
        for (file, path) in model["files"].items():
            if path is not None:
                self.files[file] = str(parent / path)


        self.arguments = {}
        for argument in [
            profiles_json["defaults"],
            profiles_json[self.profile],
            self.parameters,
            self.files
            ]:
            
            self.arguments.update(argument)         


    def build_command(self) -> list:
        """ Build a valid Popen list to start a llama.cpp using the model's self arguments  """
        command = ["llama-server"]
        for parameter, value in self.parameters.items():
            command.append(parameter)
            if value != "":
                command.append(str(value))
                      
        return command



class Loader:
    def __init__(self, args: argparse.Namespace):
        self.models = {}  # Dictionary with the models available 
        self.args = args  # Arguments from the argparser
        self.configs = json.load((ROOT / Path("configs.json")).open("r", encoding='utf-8'))
        self.profiles_json = json.load((ROOT / Path("profiles.json")).open("r", encoding='utf-8')) 
        

        models_path = Path(self.configs["models_paths"])
        json_paths = models_path.rglob("*.json")

        for json_path in json_paths:  # Construction of the models dictionary
            model_json = json.load(json_path.open("r", encoding='utf-8'))
            if {"name", "files", "alias"} <= model_json.keys():
                self.models[model_json["alias"]] = Model(model_json, json_path, json_path.parent, self.profiles_json)


    def start(self, model:str, llamaargs: list | None = None, b: bool = False, i: bool = False, a: bool = False):
        assert model in self.models.keys(), ("Not a valid model.")  # Assert that the model alias is valid

        model = self.models[model]  # Grabs the correct model
        arguments = model.arguments # And its arguments mutable

        if llamaargs:  # If there are available flags
            profile_arg = llamaargs[0]  # collect the first one
        
            if profile_arg in self.profiles_json.keys():  # If the first flag is a profile name
                llamaargs.pop(0)  # IF IT'S A FLAG WE POP IT! WE NEED THIS LIST TO CONTAIN ONLY llama.cpp flags! We POP directly the mutable, it's not a bug, it's a feature
                
                profile_json = self.profiles_json[profile_arg]  # Create a copy of the profile
                arguments.update(profile_json)  # Update the arguments with the profile selected by the user
            
            elif not profile_arg.startswith("-"):
                raise NameError("Not a valid profile or llama.cpp.")
                
            flags_dict = Argparser.args_to_dict(llamaargs)  # Create a dictionary with the llama.cpp flags
            arguments.update(flags_dict)  # Update the model's arguments
                
        command = model.build_command()  # Create a Popen command with everything! Note that, it just updates if the user insert flags or select a profile


        if b or i:  # Check if the browser or incognito flag is set
            self.open_browser(arguments, i)  # If it is, then open the browser. NOTE: The llama-ui WAITS for the model to load, this is not a BUG!
        elif a:
            subprocess.Popen([self.configs["harness"], Path.cwd()])


        # Just start the server with the command we just made
        process = subprocess.Popen(command)

        try:
            process.wait()  # Waits for the process
        except KeyboardInterrupt:  # If the user uses CTRL + C
                print("\nClosing the server...")
                process.terminate()
                process.wait()
        

    def list(self, models: bool, profiles: bool):
        if models:
            print("\nModels:")
            for alias, modelo in self.models.items():
                print(f"{modelo.name}, alias: {modelo.alias}, profile: {modelo.profile},  path: {modelo.parent}")

        elif profiles:
            print("\nProfiles:")
            for profile in self.profiles_json.keys():
                if profile != "defaults":
                    print(profile)

        else:
            print("\nModels:")
            for alias, modelo in self.models.items():
                print(f"{modelo.name}, alias: {modelo.alias}, profile: {modelo.profile},  path: {modelo.parent}")

            print("\nProfiles:")
            for profile in self.profiles_json.keys():
                if profile != "defaults":
                    print(profile)


    def edit(self, file: str):
        if file in ("configs", "profiles"):
            path = ROOT / Path(f"{file}.json")

        elif file in self.models.keys():
            path = self.models[file].path

        else:
            raise NameError("Not a file or model to edit.")

        if path.is_file():
            subprocess.Popen([self.configs["editor"], path])
        else:
            raise NameError("File doesn't exists")


    def show(self, object: str, profile: str = None):
        if object in self.profiles_json.keys():
            for (key, value) in self.profiles_json[object].items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)

        elif object in self.models.keys():
            model = self.models[object]

            if profile:
                model.arguments.update(self.profiles_json[profile])

            for (key, value) in model.arguments.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
        else:
            raise NameError("Not a model or profile.")


    def run(self):
        match self.args.command:
            case "list":
                self.list(self.args.models, self.args.profiles)
            case "edit":
                self.edit(self.args.file)
            case "show":
                self.show(self.args.object, self.args.profile)
            case "start":
                self.start(self.args.model, self.args.llamaargs, self.args.b, self.args.i, self.args.a)


    def open_browser(self, arguments: dict, incognito: bool = False):        
        command = [
            Path(self.configs["browser_path"]),
            "--start-maximized",
            f"http://{arguments["--host"]}:{arguments["--port"]}"
            ]
        if incognito:
            command.append("--incognito")
            
        subprocess.Popen(command)
        


if __name__ == '__main__':
    argparser = Argparser()    
    loader = Loader(argparser.parse())

    loader.run()