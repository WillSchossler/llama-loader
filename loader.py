import tomllib
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
        self.show_parser.add_argument("target", help="TODO")
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

                if not key.startswith("-"):
                    raise ValueError(f"{key} is not a valid llama.cpp flag.")

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
    def __init__(self, model: dict, path: Path, parent: Path, profiles: dict):
        self.path = path
        self.parent = parent
        self.profiles = profiles
        self.name = model["name"]
        self.alias = model["alias"]
        self.profile = model["profile"]
        self.parameters = model["parameters"]

        self.files = {file: str(parent / path) for (file, path) in model["files"].items()}        

        self.arguments = {}
        self.build_arguments(self.profiles[self.profile])


    def build_arguments(self, profile: dict):
        """ Builds the correct arguments dict, given a specific profile """
        self.arguments.clear()

        for argument in [
            self.profiles["default"],
            profile,
            self.parameters,
            self.files
            ]:
            
            self.arguments.update(argument) 


    def build_command(self) -> list[str]:
        """ Build a valid Popen list to start a llama.cpp using the model's self arguments  """
        command = ["llama-server"]
        for parameter, value in self.arguments.items():
            command.append(parameter)
            if value != "":
                command.append(str(value))
                      
        return command



class Loader:
    def __init__(self, args: argparse.Namespace):
        self.models = {}  # Dictionary with the models available 
        self.args = args  # Arguments from the argparser
        self.configs = tomllib.load((ROOT / "configs.toml").open("rb"))        
        self.profiles = tomllib.load((ROOT / "profiles.toml").open("rb")) 
        

        models_path = Path(self.configs["models_paths"])
        toml_paths = models_path.rglob("*.toml")

        for toml_path in toml_paths:  # Construction of the models dictionary
            model_toml = tomllib.load(toml_path.open("rb"))
            if {"name", "files", "alias", "profile", "parameters"} <= model_toml.keys():
                self.models[model_toml["alias"]] = Model(model_toml, toml_path, toml_path.parent, self.profiles)


    def start(self, model:str, llamaargs: list | None = None, b: bool = False, i: bool = False, a: bool = False):
        if model not in self.models:  # Test if the model is known
            raise ValueError(f"Unknown model: {model}.")


        model = self.models[model]  # Grabs the correct model


        if llamaargs:  # If there are available flags
            llamaargs = llamaargs.copy()  # Make a copy of the list to prevent errors 
            profile_arg = llamaargs[0]    # Collect the first one
        

            if profile_arg in self.profiles:  # If the first flag is a profile name
                llamaargs.pop(0)  # IF IT'S A FLAG WE POP IT! WE NEED THIS LIST TO CONTAIN ONLY llama.cpp flags! We POP directly the mutable, it's not a bug, it's a feature
                
                new_profile = self.profiles[profile_arg]  # Catch the correct new profile
                model.build_arguments(new_profile)  # Update the arguments with the profile selected by the user
            

            elif not profile_arg.startswith("-"):
                raise ValueError(f"{profile_arg} is not a valid profile or llama.cpp flag.")

            flags_dict = Argparser.args_to_dict(llamaargs)  # Create a dictionary with the llama.cpp flags
            model.arguments.update(flags_dict)  # Updates the model with the CLI flags

        command = model.build_command()  # Create a Popen command with everything! Note that, it just updates if the user insert flags or select a profile

        if b or i:  # Check if the browser or incognito flag is set
            self.open_browser(model.arguments, i)  # If it is, then open the browser. NOTE: The llama-ui WAITS for the model to load, this is not a BUG!
        elif a:
            subprocess.Popen([self.configs["harness"], Path.cwd()])


        # Just start the server with the command we just made
        process = subprocess.Popen(command)

        try:
            process.wait()
        except KeyboardInterrupt:  # If the user uses CTRL + C
                print("\nClosing the server...")
                process.terminate()
                process.wait()
        

    def list(self, models: bool, profiles: bool):
        if models:
            print("\nModels:")
            for alias, modelo in self.models.items():
                print(f"alias: {modelo.alias:<10}||  name: {modelo.name:<25}||  profile: {modelo.profile:>10}  ||   path: {str(modelo.parent):<70}")

        elif profiles:
            print("\nProfiles:")
            for profile in self.profiles.keys():
                if profile != "default":
                    print(profile)

        else:
            print("\nModels:")
            for alias, modelo in self.models.items():
                print(f"alias: {modelo.alias:<10}||  name: {modelo.name:<25}||  profile: {modelo.profile:>10}  ||   path: {str(modelo.parent):<70}")

            print("\nProfiles:")
            for profile in self.profiles.keys():
                if profile != "default":
                    print(profile)


    def edit(self, file: str):
        if file in ("configs", "profiles"):
            path = ROOT / Path(f"{file}.toml")

        elif file in self.models:
            path = self.models[file].path

        else:
            raise ValueError(f"{file} is not a valid model or file.")

        if path.is_file():
            subprocess.Popen([self.configs["editor"], path])
        else:
            raise FileNotFoundError(f"{file} doesn't exists.")


    def show(self, target: str, profile: str | None = None):
        if target in self.profiles.keys():
            for (key, value) in self.profiles[target].items():
                if value != "":
                    print(f"{key}: {value}")
                else:
                    print(key)

        elif target in self.models:
            model = self.models[target]

            if profile:
                if profile not in self.profiles:
                    raise ValueError(f"Unknown profile: {profile}")

                else:
                    model.build_arguments(self.profiles[profile])                

            for (key, value) in model.arguments.items():
                if value != "":
                    print(f"{key}: {value}")
                else:
                    print(key)

        else:
            raise ValueError(f"{target} is not a valid model or profile.")


    def run(self):
        match self.args.command:
            case "list":
                self.list(self.args.models, self.args.profiles)
            case "edit":
                self.edit(self.args.file)
            case "show":
                self.show(self.args.target, self.args.profile)
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