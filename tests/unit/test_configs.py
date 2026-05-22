import json
from pathlib import Path

from configs.schema import validate_config


def test_all_existing_configs_validate() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    for cfg_path in config_dir.rglob("*.json"):
        if cfg_path.name == "dataset.json":
            continue
        with cfg_path.open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        validated = validate_config(cfg)
        assert validated.training.epochs > 0
        assert validated.training.batch_size > 0
