import argparse
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent



class Argparser:
    """ TODO: Describe the class """
    def __init__(self):
        # First definitions used to create the arguments
        self.parser = argparse.ArgumentParser(description="TODO", prog="llama-loader")
        self.subparser = self.parser.add_subparsers(dest="command", required=True, help="TODO")


        # Create a draft .toml in the CWD. The result .toml will be filled if the files in the folder are appropriately named
        self.init_parser = self.subparser.add_parser("init")


        # Edit the model's TOML file. Use the "editor" set in configs.toml to open the file
        self.edit_parser = self.subparser.add_parser("edit")
        self.edit_parser.add_argument("file", help="TODO")


        # Show model's parameters. Pick a model or use the option "profile" flag to show the final result
        self.show_parser = self.subparser.add_parser("show")
        self.show_parser.add_argument("target", help="TODO")
        self.show_parser.add_argument("profile", nargs="?", help="TODO")


        # List models and/or profiles. Choose between "-m" or "-p" optional flags to filter the result
        self.list_parser = self.subparser.add_parser("list")
        list_group = self.list_parser.add_mutually_exclusive_group()
        list_group.add_argument("--models", "-m", action="store_true", help="TODO")
        list_group.add_argument("--profiles", "-p", action="store_true", help="TODO")


        # Start the llama.cpp server.
        self.start_parser = self.subparser.add_parser("start")
        self.start_parser.add_argument("model", help="TODO")        
        # Use the optional flags "-b", "-i" or "-a" BEFORE  <model> to open the browser normally, incognito, or start the harness
        start_group = self.start_parser.add_mutually_exclusive_group()
        start_group.add_argument("-b", action='store_true', help="TODO")
        start_group.add_argument("-i", action='store_true', help="TODO")
        start_group.add_argument("-a", action='store_true', help="TODO")
        # You can choose a profile after the <model> and/or pick as many llama.cpp flags as you want. Those have maximum priority
        self.start_parser.add_argument("llamaargs", nargs=argparse.REMAINDER, help="TODO")


    @staticmethod
    def args_to_dict(args: list[str]) -> dict[str, str]:
            """ Converts the "llamaargs" list into a valid dictionary. Flags without value will produce a {flag: ""} item  """
            def is_flag(value: str) -> bool:
                """ Auxiliary parser used to check if the argument is a flag or a value """
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
    """ TODO: Describe the class """
    def __init__(self, args: argparse.Namespace):
        self.models = {}  # Keeps models in a dict {name: Model}
        self.args = args  # Arguments collected from the argparser
        self.configs = Configs(ROOT / "configs.toml")    # Configurations set in the configs.toml       
        self.profiles = tomllib.load((ROOT / "profiles.toml").open("rb"))  # Profiles defined by the user


        # Populate self.models dictionary. Looks for any ".toml" file in the root set in the configs.toml 
        models_root = Path(self.configs.root)
        toml_paths = models_root.rglob("*.toml")
        required_fields = {"name", "files", "profile", "parameters"}

        for toml_path in toml_paths:
            model_toml = tomllib.load(toml_path.open("rb"))
            
            # Will consider a valid model ONLY if the .toml have a name, file, profile and parameter set
            if required_fields <= model_toml.keys():
                name = model_toml["name"]
                
                if name in self.models:  # Validation for the duplicate "name" case
                    raise ValueError(f'Invalid model at "{toml_path}". The name "{name}" already exists.')                
                else:
                    self.models[model_toml["name"]] = Model(model_toml, toml_path, toml_path.parent, self.profiles)


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
            browser = self.configs.require_browser()
            self.open_browser(browser, model.arguments["host"], model.arguments["port"], i)  # If it is, then open the browser. NOTE: The llama-ui WAITS for the model to load, this is not a BUG!
        elif a:
            subprocess.Popen([self.configs.harness, Path.cwd()])


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
            for model in self.models.values():
                print(f"Name: {model.name:<10}||  Profile: {model.profile:>10}  ||   Path: {model.parent.resolve()!s:<70}")

        elif profiles:
            print("\nProfiles:")
            for profile in self.profiles:
                if profile != "default":
                    print(profile)

        else:
            print("\nModels:")
            for model in self.models.values():
                print(f"Name: {model.name:<10}||  Profile: {model.profile:>10}  ||   Path: {model.parent.resolve()!s:<70}")

            print("\nProfiles:")
            for profile in self.profiles:
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
            subprocess.Popen([self.configs.editor, path])
        else:
            raise FileNotFoundError(f"{file} doesn't exists.")


    def show(self, target: str, profile: str | None = None):
        if target in self.profiles:
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


    def open_browser(self, browser_path, host="127.0.0", port="9993", incognito: bool = False) -> None:
        """ Opens the browser set in configs.toml. You can choose to open in incognito """        
        command = [browser_path, "--start-maximized", f"http://{host}:{port}"]
        if incognito:
            command.append("--incognito")
            
        subprocess.Popen(command)
        


class Configs:
    """ TODO: Describe class """    
    def __init__(self, path: Path):  
        with path.open("rb") as file:
            data = tomllib.load(file)

        self.root = self.__validate(field="root", field_type="dir", data=data)
        self.editor = self.__validate(field="editor", field_type="str", data=data)
        self.harness = self.__validate(field="harness", field_type="str", data=data, required=False)
        self.browser_path = self.__validate(field="browser_path", field_type="file", data=data, required=False)


    def require_browser(self) -> Path:
        """ Returns the browser's path if available """
        if self.browser_path is None:
            raise ValueError("Browser is not configured")

        return self.browser_path


    def __validate(self, field: str, field_type: str, data: dict, required: bool = True):
        """ TODO: Describe validation """

        # Check if the field is defined
        if field not in data:
            if required:
                raise ValueError(f"Field '{field}' not defined in configs.toml")
            # Optional field: absence is represented by None
            return None 

        # Colect the value of the field
        value = data[field]

        # Check if the value is a string
        if not isinstance(value, str):
                raise TypeError(f"Field '{field}' must be a string. Got '{value}' ({type(value)})")
        
        # Check if the string is empty
        if not value.strip():
            raise ValueError(f"Field '{field}' cannot be empty")


        # Check field's type
        match field_type:
            case "dir":
                path = Path(value)
                if path.is_dir():
                    return path
                else:
                    raise ValueError(f"Value '{value}' is not a valid '{field}' directory")

            case "file":
                path = Path(value)
                if path.is_file():
                    return path
                else:
                    raise ValueError(f"Field '{field}' does not contain a valid file ({value})")

            case "str":
                return value

            case _:
                raise ValueError(f"Unknown validation type '{field_type}'")



        
        



if __name__ == '__main__':
    argparser = Argparser()    
    loader = Loader(argparser.parser.parse_args())
    loader.run()