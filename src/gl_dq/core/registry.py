"""Check plugin registry. Decorate a Check subclass with @register_check("name")."""
from __future__ import annotations

import importlib
import pkgutil

CHECKS: dict[str, type] = {}


def register_check(name: str):
    def deco(cls):
        if name in CHECKS and CHECKS[name] is not cls:
            raise ValueError(f"check {name!r} registered twice ({CHECKS[name]} vs {cls})")
        cls.name = name
        CHECKS[name] = cls
        return cls

    return deco


def discover(extra_modules: list[str] | None = None) -> dict[str, type]:
    """Import every module in gl_dq.checks (and configured plugin modules)."""
    import gl_dq.checks as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if not mod.name.startswith("_") and mod.name != "base":
            importlib.import_module(f"{pkg.__name__}.{mod.name}")
    for m in extra_modules or []:
        importlib.import_module(m)
    return dict(sorted(CHECKS.items(), key=lambda kv: kv[1].default_order))
