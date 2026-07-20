import json
from pathlib import Path
from typing import Any


CONFIG_PATH = Path("config/voice_config.json")


def load_voice_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Voice config not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)
