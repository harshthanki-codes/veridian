from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _parse_csv(raw_value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not raw_value:
        return default
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    return tuple(values) if values else default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    model_dir: Path
    storage_path: Path
    tenant_config_path: Path
    allowed_origins: tuple[str, ...]
    default_batch_limit: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    return Settings(
        project_root=project_root,
        model_dir=project_root / "models",
        storage_path=Path(
            os.getenv("VERIDIAN_STORAGE_PATH", project_root / "data" / "veridian_runtime.db")
        ),
        tenant_config_path=Path(
            os.getenv("VERIDIAN_TENANTS_PATH", project_root / "config" / "tenants.json")
        ),
        allowed_origins=_parse_csv(os.getenv("VERIDIAN_ALLOWED_ORIGINS"), ("*",)),
        default_batch_limit=int(os.getenv("VERIDIAN_DEFAULT_BATCH_LIMIT", "500")),
    )
