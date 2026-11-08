"""Experiment entry point for the TinyFL benchmark."""

from dataclasses import fields
from pathlib import Path
from typing import Any

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from ddosdetTinyFL.experimental_config import ExperimentalConfig, get_default_config
from ddosdetTinyFL.experiment_runner import run_experiment

app = ServerApp()

def _load_config(run_config: Any) -> ExperimentalConfig:
    config = get_default_config()
    config_path = run_config.get("experiment-config")
    if config_path:
        path = Path(str(config_path)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Experiment config not found: {path}")
        config = ExperimentalConfig.from_json(str(path))

    valid_fields = {field.name for field in fields(ExperimentalConfig)}
    for key, value in run_config.items():
        attr = str(key).replace("-", "_")
        if attr in valid_fields:
            setattr(config, attr, value)
    return config

@app.main()
def main(grid: Grid, context: Context) -> None:
    config = _load_config(context.run_config)
    results = run_experiment(config, grid)
    print(results)