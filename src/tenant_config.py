from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

from src.settings import get_settings


@dataclass(frozen=True)
class TenantConfig:
    tenant_id: str
    display_name: str
    api_key: str
    decision_threshold: float = 0.5
    batch_limit: int = 500
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TenantCatalog:
    tenants_by_id: dict[str, TenantConfig]
    tenants_by_api_key: dict[str, TenantConfig]

    @property
    def is_configured(self) -> bool:
        return bool(self.tenants_by_id)

    def by_api_key(self, api_key: str) -> TenantConfig | None:
        return self.tenants_by_api_key.get(api_key)


def _build_tenant_catalog(items: list[dict]) -> TenantCatalog:
    tenants_by_id: dict[str, TenantConfig] = {}
    tenants_by_api_key: dict[str, TenantConfig] = {}

    for item in items:
        tenant = TenantConfig(
            tenant_id=item["tenant_id"],
            display_name=item.get("display_name", item["tenant_id"]),
            api_key=item["api_key"],
            decision_threshold=float(item.get("decision_threshold", 0.5)),
            batch_limit=int(item.get("batch_limit", 500)),
            tags=tuple(item.get("tags", [])),
        )
        tenants_by_id[tenant.tenant_id] = tenant
        tenants_by_api_key[tenant.api_key] = tenant

    return TenantCatalog(tenants_by_id=tenants_by_id, tenants_by_api_key=tenants_by_api_key)


@lru_cache(maxsize=1)
def get_tenant_catalog() -> TenantCatalog:
    settings = get_settings()

    if settings.tenant_config_path.exists():
        payload = json.loads(settings.tenant_config_path.read_text(encoding="utf-8"))
        return _build_tenant_catalog(payload.get("tenants", []))

    default_api_key = os.getenv("VERIDIAN_DEFAULT_API_KEY")
    if not default_api_key:
        return TenantCatalog(tenants_by_id={}, tenants_by_api_key={})

    fallback_tenant = {
        "tenant_id": os.getenv("VERIDIAN_DEFAULT_TENANT_ID", "default"),
        "display_name": os.getenv("VERIDIAN_DEFAULT_TENANT_NAME", "Default Tenant"),
        "api_key": default_api_key,
        "decision_threshold": float(os.getenv("VERIDIAN_DEFAULT_THRESHOLD", "0.5")),
        "batch_limit": int(os.getenv("VERIDIAN_DEFAULT_BATCH_LIMIT", "500")),
    }
    return _build_tenant_catalog([fallback_tenant])
