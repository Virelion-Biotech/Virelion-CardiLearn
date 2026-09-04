"""Load and validate versioned YAML experiment configurations."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a reproducibility configuration is invalid."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without silently accepting a non-mapping root."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise ConfigError("YAML configs require PyYAML; install the 'bench' extra") from exc
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("configuration root must be a mapping")
    return payload


def require_keys(mapping: dict[str, Any], keys: tuple[str, ...], *, section: str = "config") -> None:
    """Require named keys while preserving the full user configuration."""
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"{section} missing required keys: {', '.join(missing)}")


def validate_reproducibility_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the minimum contract for a reproducible CardiLearn experiment."""
    require_keys(config, ("schema_version", "experiment", "data", "splits", "reproducibility"))
    if str(config["schema_version"]) != "1.0":
        raise ConfigError("unsupported reproducibility config schema_version; expected '1.0'")
    experiment = config["experiment"]
    data = config["data"]
    splits = config["splits"]
    repro = config["reproducibility"]
    for section_name, section in (("experiment", experiment), ("data", data), ("splits", splits), ("reproducibility", repro)):
        if not isinstance(section, dict):
            raise ConfigError(f"{section_name} must be a mapping")
    require_keys(experiment, ("name", "task"), section="experiment")
    require_keys(data, ("source", "fingerprint"), section="data")
    require_keys(splits, ("manifest", "group_column"), section="splits")
    require_keys(repro, ("seeds", "primary_seed", "fit_preprocessing_on_train_only"), section="reproducibility")
    seeds = repro["seeds"]
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) for seed in seeds):
        raise ConfigError("reproducibility.seeds must be a non-empty list of integers")
    if repro["primary_seed"] not in seeds:
        raise ConfigError("reproducibility.primary_seed must appear in seeds")
    if repro["fit_preprocessing_on_train_only"] is not True:
        raise ConfigError("fit_preprocessing_on_train_only must be true")
    if not isinstance(data["fingerprint"], str) or len(data["fingerprint"]) != 64:
        raise ConfigError("data.fingerprint must be a SHA256 hex digest")
    return config
