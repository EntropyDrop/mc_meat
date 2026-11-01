"""Resolve machine-specific dependencies without storing private paths in Git."""
import json
import os
from pathlib import Path


def dependency_path(key, default):
    """Environment overrides user configuration; relative JSON paths use its directory."""
    override = os.environ.get('MC_MEAT_' + key.upper())
    if override:
        return Path(override).expanduser().resolve()
    explicit = os.environ.get('MC_MEAT_CONFIG')
    config = Path(explicit).expanduser() if explicit else Path.home() / '.config/mc_meat/paths.json'
    if explicit or config.exists():
        values = json.loads(config.read_text())
        if not isinstance(values, dict):
            raise ValueError('MC Meat path configuration must be a JSON object')
        if key in values:
            value = values[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError('MC Meat dependency paths must be nonempty strings')
            path = Path(value).expanduser()
            return (path if path.is_absolute() else config.parent / path).resolve()
    return Path(default).expanduser().resolve()
