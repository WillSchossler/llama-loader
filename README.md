# llama-loader

A small Python script to load [llama.cpp](https://github.com/ggml-org/llama.cpp) models via the terminal.
Models, profiles and UI settings are all described in JSON files, so you can switch models without
rebuilding the command line.

## How it works

Every model is a single `.json` file somewhere under `models_paths` (see `configs.json`).
When you run `start`, settings are merged in this priority order (lowest to highest):

1. `profiles.json` -> `defaults`
2. The profile selected by the model (`profile` field)
3. The model's own `parameters`
4. The model's `files` (resolved relative to the model's folder)
5. Extra `llama.cpp` flags passed on the command line

The resulting list is passed to `llama-server`. `llamaui.json` is injected into the server via
`--ui-config`; if the profile defines a `systemMessage`, it overrides the one in `llamaui.json`.

## Requirements

- Windows
- Python 3.10+
- `llama-server` (llama.cpp) available on `PATH`
- VS Code on `PATH` (only needed for the `edit` command)

## Files

| File           | Purpose                                                        |
| -------------- | -------------------------------------------------------------- |
| `loader.py`    | The whole loader                                               |
| `loader.cmd`   | Convenience wrapper: `loader.cmd list`, `loader.cmd start ...` |
| `configs.json` | Paths: where models live, browser, `profiles.json`, `llamaui.json` |
| `profiles.json`| Flag presets: one `defaults` block plus named profiles         |
| `llamaui.json` | LLM UI settings, injected into the server via `--ui-config`    |

## Model JSON

Each model is a JSON file anywhere under `models_paths` (subfolders are allowed).
Files are resolved relative to the folder containing the JSON:

```json
{
    "name": "Qwen3.8-27B",
    "alias": "qwen",
    "profile": "xhigh",

    "files": {
        "--model": "Qwen3.8-27B-Dense-UD-IQ3_XXS.gguf",
        "--mmproj": "Qwen3.8-27B-mmproj-BF16.gguf",
        "--model-draft": null,
        "--chat-template-file": "Qwen3.8-27B-Template.jinja"
    },

    "parameters": {
        "--cache-type-k": "q8_0"
    }
}
```

- `alias` - short name used on the command line
- `profile` - key from `profiles.json` (e.g. `xhigh`)
- `files` - maps llama.cpp flags to files in the same folder; set a value to `null` to skip that flag entirely
- `parameters` - any extra llama.cpp flags for this model

## profiles.json

```json
{
    "defaults": {
        "--port": 9931,
        "--host": "127.0.0.1",
        "--fit": "on",
        "--threads": 10
    },

    "xhigh": {
        "systemMessage": null, 

        "--temp": 1.00,
        "--top-p": 0.95,
        "--reasoning": "on",
        "--reasoning-effort": "xhigh"
    },

    "off": {
        "systemMessage": null, 

        "--temp": 1.00,
        "--reasoning": "off"
    }
}
```

- `defaults` is merged into every model
- each model picks one named profile via its `profile` field
- a profile may include `systemMessage`, which overrides `llamaui.json`'s one (use `null` to keep the global one)
- an empty string value (`"--jinja": ""`) passes the flag as a value-less flag

## Usage

All commands work through `python loader.py ...` or the `loader.cmd` wrapper.

### list

Show every available model:

```
loader list
```

### edit

Open a JSON file in VS Code - any model, or the global configs:

```
loader edit qwen      # model with alias "qwen"
loader edit configs   # configs.json
loader edit profiles
loader edit llamaui
```

### start

Start `llama-server` for a model. You can append ANY valid llama.cpp flag - they have the
highest priority and can override defaults, profile and model values:

```
loader start qwen
loader start qwen --cache-type-k q8_0
loader start -b gemma
loader start -bi gemma
```

| Flag  | Meaning                                                      |
| ----- | ------------------------------------------------------------ |
| `-b`  | Open the browser (path from `configs.json`) at the server URL |
| `-i`  | Open the browser in incognito (combine with `-b`)            |

Flags on the command line are parsed as `--flag value` pairs; a bare `--flag` (no value) is
also valid.

## configs.json

```json
{
    "models_paths": "F:/AI/LLMs",
    "browser_path": "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe",
    "llamaui_path": "llamaui.json",
    "profiles_path": "profiles.json"
}
```

- `models_paths` - absolute path, scanned recursively for model `*.json` files
- `browser_path` - browser executable used by `-b` / `-i`
- `llamaui_path` / `profiles_path` - relative to this folder