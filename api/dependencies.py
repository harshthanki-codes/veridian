from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, Request

from src.storage import RuntimeStore, get_store
from src.tenant_config import TenantConfig, get_tenant_catalog


@dataclass(frozen=True)
class RequestContext:
    tenant: TenantConfig
    request_id: str


def get_runtime_store() -> RuntimeStore:
    return get_store()


def get_request_context(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> RequestContext:
    catalog = get_tenant_catalog()
    if not catalog.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Tenant configuration not loaded. Configure VERIDIAN_TENANTS_PATH or VERIDIAN_DEFAULT_API_KEY.",
        )

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    tenant = catalog.by_api_key(x_api_key)
    if tenant is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    request.state.tenant = tenant
    return RequestContext(tenant=tenant, request_id=request.state.request_id)
