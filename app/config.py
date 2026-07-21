import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "voice_config.json"


def get_repo_root() -> Path:
    return REPO_ROOT


def resolve_repo_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_voice_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Voice config not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)
